"""Справочник категорий, политик и порогов. Единая точка правды для PoC."""

from __future__ import annotations

from typing import Dict, List

# Версии артефактов пишутся в аудит-лог: по ним восстанавливается,
# какая именно логика приняла решение.
RULES_VERSION = "rules-2026.08.16"
CLASSIFIER_VERSION = "tfidf-knn-0.1.0"
POLICY_VERSION = "policy-2026.08.16"

CATEGORIES: List[str] = [
    "delivery_status",
    "product_question",
    "promo_code",
    "cancel_order",
    "return_refund",
    "payment_issue",
    "account_access",
    "complaint_other",
]

# Псевдокатегория для случая, когда во входном тексте нет ни одного сигнала:
# ни ключевых слов, ни похожих размеченных примеров. Записывать в аудит-лог
# конкретную тему в такой ситуации нельзя — система её не определила.
UNKNOWN_CATEGORY = "unknown"

CATEGORY_RU: Dict[str, str] = {
    "unknown": "Тема не определена",
    "delivery_status": "Статус доставки",
    "product_question": "Вопрос о товаре",
    "promo_code": "Промокоды и скидки",
    "cancel_order": "Отмена заказа",
    "return_refund": "Возврат товара и денег",
    "payment_issue": "Проблема с оплатой",
    "account_access": "Доступ к аккаунту",
    "complaint_other": "Жалоба и прочее",
}

# Категории, которые РАЗРЕШЕНО закрывать автоматически (без оператора).
# Всё, что не в этом списке, максимум доходит до режима suggest (черновик оператору).
AUTO_CLOSE_ALLOWED = {"delivery_status", "product_question"}

# Категории, которые НИКОГДА не закрываются автоматически: деньги, безопасность,
# юридические последствия. Обоснование - в docs/risks-and-ops.md.
NEVER_AUTO_CLOSE = {
    "return_refund",
    "payment_issue",
    "account_access",
    "complaint_other",
}

# Очередь оператора по категории (маршрутизация).
QUEUE_BY_CATEGORY: Dict[str, str] = {
    "delivery_status": "logistics",
    "product_question": "catalog",
    "promo_code": "billing",
    "cancel_order": "orders",
    "return_refund": "finance",
    "payment_issue": "finance",
    "account_access": "security",
    "complaint_other": "escalations",
}

# Ключевые слова правил: токены уже обрезаны до 5 символов (см. textutil.tokenize).
RULE_KEYWORDS: Dict[str, List[str]] = {
    "delivery_status": ["доста", "курье", "трек", "посыл", "приед", "везут", "отсле", "адрес"],
    "product_question": ["разме", "матер", "харак", "цвет", "габар", "соста", "карто", "чехол"],
    "promo_code": ["промо", "купон", "скидк", "акции", "балл"],
    "cancel_order": ["отмен", "отказ", "перед"],
    "return_refund": ["возвр", "верну", "брако", "обмен"],
    "payment_issue": ["оплат", "списа", "плате", "карты", "средс"],
    "account_access": ["аккау", "парол", "войти", "взлом", "кабин", "профи", "досту"],
    "complaint_other": ["жалоб", "безоб", "ужасн", "компе", "требу", "суд"],
}

# --- Пороги принятия решений -------------------------------------------------
# CONFIDENCE_AUTO   - минимальная уверенность классификатора для автоответа.
# MARGIN_AUTO       - минимальный отрыв топ-1 категории от топ-2 (защита от
#                     ситуации «две категории по 0.45, модель монетку кинула»).
# CONFIDENCE_ROUTE  - ниже этого порога тему считаем неопределённой и отправляем
#                     тикет в общую очередь разбора, а не в профильную.
# KB_MIN_SCORE      - минимальная близость статьи базы знаний, чтобы на неё
#                     вообще можно было ссылаться в ответе.
# Значения подобраны на 15 тикетах PoC. В целевой системе калибруются на
# отложенной выборке под целевой precision, см. docs/ml.md.
CONFIDENCE_AUTO = 0.55
MARGIN_AUTO = 0.15
CONFIDENCE_ROUTE = 0.30
KB_MIN_SCORE = 0.08

# Ориентир SLA (Service Level Agreement - соглашение об уровне сервиса).
# Здесь: бюджет синхронного «горячего» пути на один тикет.
HOT_PATH_SLO_MS = 500.0
