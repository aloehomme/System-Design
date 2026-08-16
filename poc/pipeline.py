"""Оркестрация обработки тикета.

Разделение, которое повторяет целевую архитектуру (docs/architecture.md):

  ГОРЯЧИЙ ПУТЬ (синхронный, бюджет 500 мс): приём -> маскирование PII ->
  классификация темы -> оценка риска -> policy engine -> запись в аудит.
  Ответ вызывающей стороне отдаётся здесь. LLM в этом пути НЕ участвует.

  ХОЛОДНЫЙ ПУТЬ (асинхронный, секунды): поиск по базе знаний -> генерация
  черновика LLM -> отправка клиенту или постановка черновика в очередь оператору.

Из этого разделения следует главное свойство: отказ внешнего LLM API не влияет
на горячий путь, поэтому SLA по маршрутизации не нарушается никогда.
"""

from __future__ import annotations

import datetime
import time
from typing import Any, Dict, List, Optional

from .audit import AuditLog
from .classifier import TopicClassifier
from .llm import CircuitBreaker, fallback_template, generate_with_breaker
from .policy import decide
from .retrieval import KnowledgeBase, TicketHistory
from .risk import assess_risk, extract_order_ids, mask_pii
from .taxonomy import CATEGORY_RU, CONFIDENCE_ROUTE, HOT_PATH_SLO_MS


class Pipeline:
    def __init__(self, classifier: TopicClassifier, kb: KnowledgeBase,
                 history: TicketHistory, provider, audit: AuditLog,
                 breaker: Optional[CircuitBreaker] = None) -> None:
        self.classifier = classifier
        self.kb = kb
        self.history = history
        self.provider = provider
        self.audit = audit
        self.breaker = breaker if breaker is not None else CircuitBreaker()

    # --- Горячий путь --------------------------------------------------------
    def run_hot_path(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        masked = mask_pii(str(ticket["text"]))
        order_ids = extract_order_ids(masked["text"])
        timings["pii_masking"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        topic = self.classifier.predict(masked["text"])
        timings["classify_topic"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        risk = assess_risk(
            text=masked["text"],
            category=topic.category,
            customer_tier=str(ticket.get("customer_tier", "standard")),
            has_prior_contact=bool(ticket.get("has_prior_contact", False)),
            category_reliable=topic.confidence >= CONFIDENCE_ROUTE,
        )
        timings["assess_risk"] = (time.perf_counter() - t0) * 1000.0

        # Поиск по базе знаний в PoC выполняется в горячем пути, потому что он
        # дёшев (TF-IDF по 10 статьям) и его результат нужен policy engine для
        # решения об автоответе. В целевой системе это ANN-поиск (Approximate
        # Nearest Neighbours) по векторной базе, бюджет 35 мс - см. architecture.md.
        t0 = time.perf_counter()
        kb_hits = self.kb.search(masked["text"], top_k=3)
        similar = self.history.search(masked["text"], exclude_id=str(ticket["id"]), top_k=2)
        timings["retrieval"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        decision = decide(topic, risk, kb_hits)
        timings["policy_engine"] = (time.perf_counter() - t0) * 1000.0

        hot_total = sum(timings.values())
        return {
            "ticket": ticket,
            "masked": masked,
            "order_ids": order_ids,
            "topic": topic,
            "risk": risk,
            "kb_hits": kb_hits,
            "similar": similar,
            "decision": decision,
            "timings": timings,
            "hot_path_ms": hot_total,
            "sla_ok": hot_total <= HOT_PATH_SLO_MS,
        }

    # --- Холодный путь -------------------------------------------------------
    def run_cold_path(self, hot: Dict[str, Any]) -> Dict[str, Any]:
        decision = hot["decision"]
        kb_hits: List[Any] = hot["kb_hits"]
        best_kb = kb_hits[0] if kb_hits else None

        if not decision.allow_llm_draft:
            return {
                "draft": "",
                "draft_source": "none",
                "llm_ok": False,
                "llm_error": "генерация отключена политикой (правило %s)" % decision.rule_id,
                "degraded": False,
                "answer_sent": False,
            }

        result = generate_with_breaker(
            provider=self.provider,
            breaker=self.breaker,
            ticket_text=hot["masked"]["text"],
            kb_title=best_kb.title if best_kb else "",
            kb_text=best_kb.text if best_kb else "",
        )

        if result.ok and result.text and "НЕДОСТАТОЧНО ДАННЫХ" not in result.text:
            return {
                "draft": result.text,
                "draft_source": result.provider,
                "llm_ok": True,
                "llm_error": "",
                "llm_latency_ms": result.latency_ms,
                "degraded": False,
                "answer_sent": decision.action == "auto_answer",
            }

        # Деградация: LLM недоступен или отказался отвечать по статье.
        # Автоответ клиенту в этом режиме НЕ отправляется по существу - только
        # шаблонное подтверждение приёма, а тикет уходит человеку.
        template = fallback_template(
            CATEGORY_RU.get(hot["topic"].category, hot["topic"].category), decision.queue
        )
        return {
            "draft": template,
            "draft_source": "template-fallback",
            "llm_ok": False,
            "llm_error": result.error or "модель вернула НЕДОСТАТОЧНО ДАННЫХ",
            "degraded": True,
            "answer_sent": decision.action == "auto_answer",
        }

    # --- Полный проход + аудит ----------------------------------------------
    def process(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        hot = self.run_hot_path(ticket)
        cold = self.run_cold_path(hot)

        decision = hot["decision"]
        # При деградации автозакрытие запрещено: клиент получает шаблон-
        # подтверждение, а тикет всё равно попадает к оператору.
        effective_action = decision.action
        if cold["degraded"] and decision.action == "auto_answer":
            effective_action = "draft_for_operator"

        record = {
            "ticket_id": ticket["id"],
            "decided_at": datetime.datetime.utcnow().isoformat() + "Z",
            "channel": ticket.get("channel", "unknown"),
            "customer_tier": ticket.get("customer_tier", "standard"),
            "masked_text": hot["masked"]["text"],
            "pii_masked": hot["masked"]["pii_found"],
            "order_ids": hot["order_ids"],
            "topic": hot["topic"].category,
            "topic_confidence": hot["topic"].confidence,
            "topic_margin": hot["topic"].margin,
            "topic_scores": hot["topic"].scores,
            "risk_level": hot["risk"].level,
            "risk_flags": hot["risk"].flags,
            "kb_top": [{"id": h.doc_id, "score": h.score} for h in hot["kb_hits"]],
            "similar_tickets": [{"id": s.ticket_id, "score": s.score} for s in hot["similar"]],
            "action": effective_action,
            "action_before_degradation": decision.action,
            "queue": decision.queue,
            "priority": decision.priority,
            "rule_id": decision.rule_id,
            "reasons": decision.reasons,
            "decided_by": "automation",
            "draft_source": cold["draft_source"],
            "llm_ok": cold["llm_ok"],
            "llm_error": cold["llm_error"],
            "degraded": cold["degraded"],
            "answer_sent": bool(cold["answer_sent"]) and not cold["degraded"],
            "latency_ms": {
                "hot_path_total": round(hot["hot_path_ms"], 2),
                **{key: round(value, 2) for key, value in hot["timings"].items()},
            },
            "sla_hot_path_ok": hot["sla_ok"],
            "versions": {
                "classifier": hot["topic"].version,
                "rules": hot["risk"].version,
                "policy": decision.version,
                "llm_provider": cold["draft_source"],
            },
        }
        written = self.audit.write(record)
        return {"hot": hot, "cold": cold, "audit": written, "effective_action": effective_action}
