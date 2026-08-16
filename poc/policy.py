"""Policy engine: превращает предсказания в решение о действии.

Это единственное место, где принимается решение «что делать с тикетом».
Правила здесь намеренно жёсткие и читаются сверху вниз: первое сработавшее
правило выигрывает. Такой порядок делает решение объяснимым аудитору -
в лог пишется номер сработавшего правила.

Возможные действия:
  escalate_operator  - эскалация оператору с пометкой причины, автоответ запрещён
  draft_for_operator - черновик готовится, но отправляет его человек (suggest)
  auto_answer        - автоответ клиенту и автозакрытие тикета
  route_to_queue     - тема непонятна, отправляем в общую очередь разбора
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

from .retrieval import KbHit
from .risk import RiskAssessment
from .classifier import TopicPrediction
from .taxonomy import (
    AUTO_CLOSE_ALLOWED,
    CONFIDENCE_AUTO,
    CONFIDENCE_ROUTE,
    KB_MIN_SCORE,
    MARGIN_AUTO,
    NEVER_AUTO_CLOSE,
    POLICY_VERSION,
    QUEUE_BY_CATEGORY,
)


class Decision(NamedTuple):
    action: str
    queue: str
    priority: str          # normal | high
    rule_id: str           # какое правило сработало
    reasons: List[str]
    allow_llm_draft: bool  # можно ли вообще звать LLM для этого тикета
    version: str
    # Статья базы знаний, на которой строится ответ. Для автоответа её выбирает
    # policy engine (с проверкой соответствия теме), а не «первая попавшаяся».
    kb_for_answer: Optional[KbHit] = None


def pick_article_for_auto_answer(topic: TopicPrediction, kb_hits: List[KbHit]) -> Optional[KbHit]:
    """Выбрать статью, на основании которой можно дать автоответ.

    Статья обязана удовлетворять трём условиям одновременно:
      1) относиться к ТОЙ ЖЕ категории, что предсказал классификатор;
      2) быть помеченной как разрешённая для автоответа;
      3) иметь близость не ниже порога.

    Пункт 1 критичен: без него тикет с темой «статус доставки» мог получить
    автоответ по статье о таблице размеров — обе статьи разрешены для
    автоответа, и проверка только по флагу пропускала такой случай.
    """
    for hit in kb_hits:
        if (hit.category == topic.category
                and hit.auto_answer_allowed
                and hit.score >= KB_MIN_SCORE):
            return hit
    return None


def decide(topic: TopicPrediction, risk: RiskAssessment, kb_hits: List[KbHit]) -> Decision:
    reasons: List[str] = []
    best_kb = kb_hits[0] if kb_hits else None
    article = pick_article_for_auto_answer(topic, kb_hits)

    # П-1. Prompt injection: тикет карантинится, LLM не вызывается вообще.
    if "prompt_injection" in risk.flags:
        return Decision(
            action="escalate_operator",
            queue="security",
            priority="high",
            rule_id="P-1",
            reasons=["Обнаружена попытка prompt injection: обращение в карантин, "
                     "генерация ответа отключена"] + risk.reasons,
            allow_llm_draft=False,
            version=POLICY_VERSION,
        )

    # П-2. Юридические угрозы и вред здоровью: только человек, без черновика.
    if "legal_or_health" in risk.flags:
        return Decision(
            action="escalate_operator",
            queue="escalations",
            priority="high",
            rule_id="P-2",
            reasons=["Юридический риск или упоминание вреда здоровью"] + risk.reasons,
            allow_llm_draft=False,
            version=POLICY_VERSION,
        )

    # П-3. Низкая уверенность модели: тему не определили - в общий разбор.
    # Это правило стоит ВЫШЕ правил, опирающихся на категорию: если модель не
    # уверена в теме, то и производные от темы флаги риска (например
    # security_category) недостоверны, и опираться на них нельзя.
    # Текстовые детекторы (П-1, П-2) от категории не зависят и уже отработали.
    if topic.confidence < CONFIDENCE_ROUTE:
        reasons.append(
            "Уверенность %.2f ниже порога маршрутизации %.2f: тема не определена, "
            "решение принимает человек" % (topic.confidence, CONFIDENCE_ROUTE)
        )
        return Decision(
            action="route_to_queue",
            queue="general_triage",
            priority="normal",
            rule_id="P-3",
            reasons=reasons,
            allow_llm_draft=False,
            version=POLICY_VERSION,
        )

    # П-4. Высокий риск (например безопасность аккаунта): эскалация,
    # но черновик оператору готовим - он экономит время человека.
    if risk.level == "high":
        return Decision(
            action="escalate_operator",
            queue=QUEUE_BY_CATEGORY.get(topic.category, "escalations"),
            priority="high",
            rule_id="P-4",
            reasons=["Высокий уровень риска"] + risk.reasons,
            allow_llm_draft=True,
            version=POLICY_VERSION,
        )

    # П-5. Категория запрещена к автозакрытию: максимум черновик оператору.
    if topic.category in NEVER_AUTO_CLOSE:
        reasons.append("Категория «%s» в списке запрещённых к автозакрытию" % topic.category)
        return Decision(
            action="draft_for_operator",
            queue=QUEUE_BY_CATEGORY.get(topic.category, "general_triage"),
            priority="high" if risk.level == "medium" else "normal",
            rule_id="P-5",
            reasons=reasons + risk.reasons,
            allow_llm_draft=True,
            version=POLICY_VERSION,
        )

    # П-6. Автоответ: безопасная категория, низкий риск, уверенность и отрыв
    # выше порогов, найдена статья базы знаний, разрешённая для автоответа.
    auto_ready = (
        topic.category in AUTO_CLOSE_ALLOWED
        and risk.level == "low"
        and topic.confidence >= CONFIDENCE_AUTO
        and topic.margin >= MARGIN_AUTO
        and article is not None
    )
    if auto_ready:
        reasons.append(
            "Уверенность %.2f >= %.2f, отрыв %.2f >= %.2f, риск low, "
            "статья %s той же категории «%s» (близость %.2f) разрешена для автоответа"
            % (topic.confidence, CONFIDENCE_AUTO, topic.margin, MARGIN_AUTO,
               article.doc_id, article.category, article.score)
        )
        return Decision(
            action="auto_answer",
            queue=QUEUE_BY_CATEGORY.get(topic.category, "general_triage"),
            priority="normal",
            rule_id="P-6",
            reasons=reasons,
            allow_llm_draft=True,
            version=POLICY_VERSION,
            kb_for_answer=article,
        )

    # П-7. Всё остальное: черновик оператору с объяснением, чего не хватило.
    if topic.category not in AUTO_CLOSE_ALLOWED:
        reasons.append("Категория «%s» не входит в белый список автозакрытия" % topic.category)
    if topic.confidence < CONFIDENCE_AUTO:
        reasons.append("Уверенность %.2f ниже порога автоответа %.2f"
                       % (topic.confidence, CONFIDENCE_AUTO))
    if topic.margin < MARGIN_AUTO:
        reasons.append("Отрыв от второй категории %.2f ниже порога %.2f"
                       % (topic.margin, MARGIN_AUTO))
    if best_kb is None or best_kb.score < KB_MIN_SCORE:
        reasons.append("Не найдена достаточно релевантная статья базы знаний")
    elif article is None:
        if not best_kb.auto_answer_allowed:
            reasons.append("Статья %s помечена как запрещённая для автоответа" % best_kb.doc_id)
        else:
            reasons.append(
                "Ближайшая статья %s относится к категории «%s», а тема тикета — «%s»: "
                "отвечать по статье из другой темы нельзя"
                % (best_kb.doc_id, best_kb.category, topic.category)
            )
    if risk.level == "medium":
        reasons.extend(risk.reasons)

    return Decision(
        action="draft_for_operator",
        queue=QUEUE_BY_CATEGORY.get(topic.category, "general_triage"),
        priority="normal",
        rule_id="P-7",
        reasons=reasons,
        allow_llm_draft=True,
        version=POLICY_VERSION,
    )
