"""Маскирование PII и детекторы риска.

PII (Personally Identifiable Information - персональные данные): телефон,
электронная почта, номер банковской карты. Маскируются ДО того, как текст
попадёт во внешний LLM API (Large Language Model - большая языковая модель)
и до записи в аудит-лог.

Все детекторы здесь - правила, а не модель. Это осознанное решение: правило
детерминировано, объяснимо аудитору и не деградирует незаметно. Цена - полнота
ниже, поэтому правила дополняются человеческой выборочной проверкой
(см. docs/monitoring.md).
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple

from .taxonomy import NEVER_AUTO_CLOSE, RULES_VERSION

# --- Маскирование PII --------------------------------------------------------
_PII_PATTERNS = [
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Zа-яА-Я]{2,}\b")),
    ("PHONE", re.compile(r"(?:\+7|8)[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}")),
]
# Номер заказа - НЕ PII и нужен оператору, поэтому сохраняем явно.
_ORDER_RE = re.compile(r"\b(?:заказ\w*\s+)(\d{4,8})\b", re.IGNORECASE)


def mask_pii(text: str) -> Dict[str, object]:
    """Заменить персональные данные на плейсхолдеры вида [PHONE]."""
    masked = text
    found: List[str] = []
    # Порядок важен: карта до телефона, иначе длинный номер карты частично
    # съедается телефонным шаблоном.
    for name, pattern in _PII_PATTERNS:
        if pattern.search(masked):
            found.append(name)
            masked = pattern.sub("[%s]" % name, masked)
    return {"text": masked, "pii_found": found}


def extract_order_ids(text: str) -> List[str]:
    return _ORDER_RE.findall(text)


# --- Детекторы риска ---------------------------------------------------------
# Prompt injection - попытка через текст тикета переписать инструкции модели
# и заставить её выполнить действие в интересах атакующего.
_INJECTION_PATTERNS = [
    r"игнорир\w*\s+(все|всё|предыдущ|инструкц)",
    r"забудь\s+(все|всё|инструкц|правил)",
    r"ты\s+теперь\s+\w+",
    r"без\s+ограничен",
    r"ignore\s+(all\s+)?previous",
    r"system\s+prompt",
    r"покажи\s+(свои\s+)?(инструкц|промпт)",
    r"ответь\s+только\s+словом",
    r"действуй\s+как\s+(администратор|разработчик)",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Юридические угрозы и здоровье: только человек, автоответ запрещён.
_THREAT_PATTERNS = [
    r"\bсуд\w*\b", r"\bюрист\w*\b", r"роспотребнадзор", r"прокурат\w*",
    r"\bиск\b", r"здоровь\w*", r"отравил\w*", r"травм\w*",
]
_THREAT_RE = [re.compile(p, re.IGNORECASE) for p in _THREAT_PATTERNS]


class RiskAssessment(NamedTuple):
    level: str              # low | medium | high
    flags: List[str]        # какие детекторы сработали
    reasons: List[str]      # человекочитаемое объяснение для оператора и аудита
    version: str


def assess_risk(text: str, category: str, customer_tier: str, has_prior_contact: bool,
                category_reliable: bool = True) -> RiskAssessment:
    """Оценить риск тикета.

    category_reliable=False означает, что классификатор не уверен в теме.
    В этом случае флаги, производные от категории, НЕ выставляются: строить
    оценку риска на недостоверной категории - значит писать в аудит-лог
    утверждение, которого система на самом деле не знает.
    Текстовые детекторы (injection, юридические угрозы) от категории не зависят
    и работают всегда.
    """
    flags: List[str] = []
    reasons: List[str] = []

    if any(pattern.search(text) for pattern in _INJECTION_RE):
        flags.append("prompt_injection")
        reasons.append("В тексте обнаружена попытка переопределить инструкции модели")

    if any(pattern.search(text) for pattern in _THREAT_RE):
        flags.append("legal_or_health")
        reasons.append("Упомянуты юридические инстанции или вред здоровью")

    if category_reliable and category == "account_access":
        flags.append("security_category")
        reasons.append("Категория связана с безопасностью аккаунта")

    if category_reliable and category in NEVER_AUTO_CLOSE:
        flags.append("restricted_category")
        reasons.append("Категория запрещена к автозакрытию (деньги/безопасность/юридические риски)")

    if customer_tier == "vip":
        flags.append("vip_customer")
        reasons.append("VIP-клиент: повышенная цена ошибки")

    if has_prior_contact:
        flags.append("repeat_contact")
        reasons.append("Повторное обращение: предыдущий ответ не решил проблему")

    # Уровень риска: высокий, если сработал хотя бы один «жёсткий» детектор.
    hard = {"prompt_injection", "legal_or_health", "security_category"}
    if hard.intersection(flags):
        level = "high"
    elif flags:
        level = "medium"
    else:
        level = "low"

    return RiskAssessment(level=level, flags=flags, reasons=reasons, version=RULES_VERSION)
