"""Демонстрационный прогон PoC: 6 сценариев + замер латентности горячего пути.

Запуск:
    python3 poc/run_demo.py                 # офлайн, детерминированно
    python3 poc/run_demo.py --provider claude   # реальный вызов Claude API
    python3 poc/run_demo.py --no-bench      # без замера латентности

Зависимости: только стандартная библиотека Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from typing import Any, Dict, List

# Позволяет запускать файл и как скрипт, и как модуль пакета.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from poc.audit import AuditLog
    from poc.classifier import TopicClassifier
    from poc.llm import CircuitBreaker, build_provider
    from poc.pipeline import Pipeline
    from poc.retrieval import KnowledgeBase, TicketHistory
    from poc.taxonomy import CATEGORY_RU, HOT_PATH_SLO_MS
else:
    from .audit import AuditLog
    from .classifier import TopicClassifier
    from .llm import CircuitBreaker, build_provider
    from .pipeline import Pipeline
    from .retrieval import KnowledgeBase, TicketHistory
    from .taxonomy import CATEGORY_RU, HOT_PATH_SLO_MS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SEED = 42

ACTION_RU = {
    "auto_answer": "АВТООТВЕТ и автозакрытие",
    "draft_for_operator": "ЧЕРНОВИК ОПЕРАТОРУ (suggest)",
    "escalate_operator": "ЭСКАЛАЦИЯ ОПЕРАТОРУ",
    "route_to_queue": "МАРШРУТИЗАЦИЯ В ОЧЕРЕДЬ РАЗБОРА",
}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hr(title: str = "") -> None:
    if title:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
    else:
        print("-" * 78)


def print_ticket(ticket: Dict[str, Any]) -> None:
    print("Тикет %s | канал: %s | тариф клиента: %s | повторное обращение: %s"
          % (ticket["id"], ticket["channel"], ticket["customer_tier"],
             "да" if ticket["has_prior_contact"] else "нет"))
    print("Текст: %s" % ticket["text"])


def print_steps(result: Dict[str, Any], show_steps: List[int]) -> None:
    hot = result["hot"]
    cold = result["cold"]
    topic = hot["topic"]
    risk = hot["risk"]
    decision = hot["decision"]
    timings = hot["timings"]

    if 1 in show_steps:
        print("\n[Шаг 1] Маскирование персональных данных + тема + риск"
              "  (%.1f мс)" % (timings["pii_masking"] + timings["classify_topic"]
                               + timings["assess_risk"]))
        print("  Маскировано PII: %s" % (", ".join(hot["masked"]["pii_found"]) or "не найдено"))
        print("  Текст после маскирования: %s" % hot["masked"]["text"][:150])
        print("  Номера заказов (не PII, нужны оператору): %s"
              % (", ".join(hot["order_ids"]) or "нет"))
        print("  Тема: %s (%s)" % (topic.category, CATEGORY_RU.get(topic.category, "")))
        print("  Уверенность: %.2f | отрыв от 2-й категории: %.2f" % (topic.confidence, topic.margin))
        top3 = sorted(topic.scores.items(), key=lambda p: -p[1])[:3]
        print("  Топ-3 категории: %s" % ", ".join("%s=%.2f" % (c, s) for c, s in top3))
        print("  Уровень риска: %s | флаги: %s"
              % (risk.level.upper(), ", ".join(risk.flags) or "нет"))
        for reason in risk.reasons:
            print("    - %s" % reason)

    if 2 in show_steps:
        print("\n[Шаг 2] Поиск в базе знаний и среди прошлых тикетов  (%.1f мс)"
              % timings["retrieval"])
        for hit in hot["kb_hits"]:
            print("  KB %s | близость %.3f | автоответ разрешён: %s | %s"
                  % (hit.doc_id, hit.score, "да" if hit.auto_answer_allowed else "НЕТ", hit.title))
        for sim in hot["similar"]:
            print("  Похожий тикет %s | близость %.3f | %s..."
                  % (sim.ticket_id, sim.score, sim.text[:60]))

    if 3 in show_steps:
        print("\n[Шаг 3] Решение policy engine  (%.1f мс)" % timings["policy_engine"])
        print("  Действие: %s" % ACTION_RU.get(result["effective_action"], result["effective_action"]))
        print("  Правило: %s | очередь: %s | приоритет: %s"
              % (decision.rule_id, decision.queue, decision.priority))
        for reason in decision.reasons:
            print("    - %s" % reason)
        if cold["degraded"]:
            print("    - ДЕГРАДАЦИЯ: генерация недоступна, автозакрытие отменено, "
                  "решение по правилу %s понижено до передачи оператору" % decision.rule_id)
        print("\n  Черновик ответа (источник: %s):" % cold["draft_source"])
        if cold["draft"]:
            for line in _wrap(cold["draft"], 72):
                print("    | %s" % line)
        else:
            print("    | (не генерировался) %s" % cold["llm_error"])
        print("  Отправлено клиенту автоматически: %s"
              % ("ДА" if result["audit"]["answer_sent"] else "НЕТ"))

    if 4 in show_steps:
        print("\n[Шаг 4] Запись решения в аудит-лог")
        record = result["audit"]
        compact = {
            "ticket_id": record["ticket_id"],
            "topic": record["topic"],
            "topic_confidence": record["topic_confidence"],
            "risk_level": record["risk_level"],
            "risk_flags": record["risk_flags"],
            "action": record["action"],
            "rule_id": record["rule_id"],
            "queue": record["queue"],
            "answer_sent": record["answer_sent"],
            "decided_by": record["decided_by"],
            "versions": record["versions"],
            "latency_ms": record["latency_ms"]["hot_path_total"],
        }
        print("  logs/decisions.jsonl <- " + json.dumps(compact, ensure_ascii=False))

    print("\n  Горячий путь суммарно: %.1f мс (бюджет %.0f мс) -> SLA %s"
          % (hot["hot_path_ms"], HOT_PATH_SLO_MS, "СОБЛЮДЁН" if hot["sla_ok"] else "НАРУШЕН"))


def _wrap(text: str, width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines


def bench_hot_path(pipeline: Pipeline, tickets: List[Dict[str, Any]], runs: int = 200) -> None:
    """Замер латентности горячего пути: утверждение «<500 мс» должно быть
    подтверждено измерением, а не заявлено."""
    samples: List[float] = []
    for i in range(runs):
        ticket = tickets[i % len(tickets)]
        result = pipeline.run_hot_path(ticket)
        samples.append(result["hot_path_ms"])
    samples.sort()

    def percentile(sorted_values: List[float], q: float) -> float:
        """q-й процентиль. Индекс ограничивается диапазоном: без этого при
        малом числе прогонов (например runs=1) выражение int(q*n)-1 давало
        отрицательный индекс и молча возвращало максимум вместо процентиля."""
        index = int(q * len(sorted_values)) - 1
        return sorted_values[min(len(sorted_values) - 1, max(0, index))]

    p50 = statistics.median(samples)
    p95 = percentile(samples, 0.95)
    p99 = percentile(samples, 0.99)
    print("  Прогонов: %d | p50 = %.2f мс | p95 = %.2f мс | p99 = %.2f мс | max = %.2f мс"
          % (runs, p50, p95, p99, samples[-1]))
    print("  Бюджет горячего пути: %.0f мс -> %s"
          % (HOT_PATH_SLO_MS, "УКЛАДЫВАЕМСЯ" if p95 <= HOT_PATH_SLO_MS else "НЕ УКЛАДЫВАЕМСЯ"))
    print("  ВАЖНО: это латентность PoC на локальных данных (10 статей базы знаний,")
    print("  40 примеров). В целевой системе бюджет расписан по этапам в docs/architecture.md.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Демо PoC обработки тикетов поддержки")
    parser.add_argument("--provider", choices=["mock", "claude"], default="mock",
                        help="mock - офлайн-заглушка (по умолчанию); claude - реальный Claude API")
    parser.add_argument("--no-bench", action="store_true", help="пропустить замер латентности")
    args = parser.parse_args()

    random.seed(SEED)

    tickets = load_jsonl(os.path.join(DATA_DIR, "tickets.jsonl"))
    kb_articles = load_jsonl(os.path.join(DATA_DIR, "kb.jsonl"))
    examples = load_jsonl(os.path.join(DATA_DIR, "labeled_examples.jsonl"))
    by_id = {t["id"]: t for t in tickets}

    classifier = TopicClassifier(examples)
    kb = KnowledgeBase(kb_articles)
    history = TicketHistory(tickets)
    audit = AuditLog()
    audit.reset()  # чистый лог на каждый запуск -> воспроизводимый вывод

    provider = build_provider(args.provider)
    pipeline = Pipeline(classifier, kb, history, provider, audit)

    print("=" * 78)
    print("PoC: автоматизация обработки тикетов поддержки маркетплейса")
    print("Провайдер генерации: %s | тикетов: %d | статей базы знаний: %d | "
          "размеченных примеров: %d" % (provider.name, len(tickets), len(kb_articles), len(examples)))
    print("=" * 78)

    # --- Сценарии 1-4: happy path, один тикет по шагам -----------------------
    hr("СЦЕНАРИИ 1-4. HAPPY PATH: тема -> поиск -> черновик -> лог решения")
    happy = by_id["T-1001"]
    print_ticket(happy)
    result = pipeline.process(happy)
    print_steps(result, show_steps=[1, 2, 3, 4])
    assert result["effective_action"] == "auto_answer", "happy path должен давать автоответ"

    # --- Сценарий 5: fallback при недоступности LLM --------------------------
    hr("СЦЕНАРИЙ 5. FALLBACK: внешний LLM API недоступен")
    print("Симулируем таймауты внешнего LLM API. Ожидаем: горячий путь не")
    print("затронут, срабатывает circuit breaker, клиент получает шаблонное")
    print("подтверждение, тикет уходит оператору, автозакрытия НЕ происходит.")
    failing_breaker = CircuitBreaker(failure_threshold=3, cooldown_s=30.0)
    failing_pipeline = Pipeline(classifier, kb, history, build_provider("failing"),
                                audit, breaker=failing_breaker)
    for ticket_id in ["T-1013", "T-1015"]:
        print()
        hr()
        print_ticket(by_id[ticket_id])
        fallback_result = failing_pipeline.process(by_id[ticket_id])
        print_steps(fallback_result, show_steps=[1, 3, 4])
        print("  Состояние предохранителя: %s (подряд отказов: %d)"
              % (failing_breaker.state, failing_breaker.consecutive_failures))
        print("  Ошибка провайдера: %s" % fallback_result["cold"]["llm_error"])
        assert fallback_result["hot"]["sla_ok"], "горячий путь обязан укладываться в SLA"
        assert not fallback_result["audit"]["answer_sent"], "в деградации автоответ запрещён"

    # --- Сценарий 6: рискованный тикет, эскалация оператору -------------------
    hr("СЦЕНАРИЙ 6. RISKY PATH: prompt injection -> эскалация оператору")
    print("Обязательный сценарий: рискованный тикет НЕ обрабатывается")
    print("автоматически, генерация ответа отключается, тикет уходит человеку")
    print("с явной причиной эскалации.")
    print()
    risky = by_id["T-1010"]
    print_ticket(risky)
    risky_result = pipeline.process(risky)
    print_steps(risky_result, show_steps=[1, 2, 3, 4])
    assert risky_result["effective_action"] == "escalate_operator"
    assert not risky_result["audit"]["answer_sent"]
    assert risky_result["audit"]["draft_source"] == "none", "LLM не должен вызываться"

    print()
    hr()
    print("Дополнительно: запрещённая к автозакрытию категория (деньги)")
    money = by_id["T-1003"]
    print_ticket(money)
    money_result = pipeline.process(money)
    print_steps(money_result, show_steps=[1, 3])
    assert money_result["effective_action"] == "draft_for_operator"

    # --- Сводка по всем тикетам ----------------------------------------------
    hr("СВОДКА ПО ВСЕМ %d ТИКЕТАМ" % len(tickets))
    print("%-8s %-18s %-6s %-8s %-26s %-8s" %
          ("ТИКЕТ", "ТЕМА", "УВЕР.", "РИСК", "ДЕЙСТВИЕ", "ОЧЕРЕДЬ"))
    hr()
    counts: Dict[str, int] = {}
    for ticket in tickets:
        outcome = pipeline.process(ticket)
        record = outcome["audit"]
        counts[record["action"]] = counts.get(record["action"], 0) + 1
        print("%-8s %-18s %-6.2f %-8s %-26s %-8s" % (
            record["ticket_id"], record["topic"], record["topic_confidence"],
            record["risk_level"], record["action"], record["queue"]))
    hr()
    total = len(tickets)
    for action, count in sorted(counts.items(), key=lambda p: -p[1]):
        print("  %-22s %2d тикетов (%.0f%%)" % (action, count, 100.0 * count / total))
    print("\n  ВАЖНО: 15 тикетов - это демонстрация работоспособности, а не оценка")
    print("  качества. Доля автоответов на реальном трафике оценивается только")
    print("  в shadow-режиме на исторических данных, см. docs/ml.md.")

    # --- Латентность ---------------------------------------------------------
    if not args.no_bench:
        hr("ЗАМЕР ЛАТЕНТНОСТИ ГОРЯЧЕГО ПУТИ")
        bench_hot_path(pipeline, tickets, runs=200)

    # --- Аудит ---------------------------------------------------------------
    hr("АУДИТ-ЛОГ")
    with open(audit.path, "r", encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    print("  Файл: logs/decisions.jsonl")
    print("  Записей: %d (каждое автоматическое решение = одна строка JSON)" % len(lines))
    print("  Проверка читаемости: %s"
          % ("все строки распарсились" if all(json.loads(l) for l in lines) else "ОШИБКА"))
    print("  Персональные данные в логе: замаскированы (поле masked_text)")
    print("\nГотово. Полный лог решений: logs/decisions.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
