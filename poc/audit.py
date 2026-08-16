"""Аудит автоматических решений.

Одно решение = одна строка JSON в logs/decisions.jsonl (формат JSON Lines:
один JSON-объект на строку, удобно читать построчно и грузить в хранилище).

Что должно быть в записи, чтобы решение можно было воспроизвести через полгода:
  - что пришло на вход (замаскированный текст, а не сырой - в логе не должно
    быть персональных данных);
  - какие версии правил/моделей/политики работали;
  - что предсказали и с какой уверенностью;
  - какое действие выбрано и по какому правилу;
  - тайминги по этапам (для разбора нарушений SLA);
  - кто принял финальное решение: automation или human.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "decisions.jsonl"
)

# Поля, обязательные в каждой записи. Проверяются тестом test_audit_schema.
REQUIRED_FIELDS = [
    "schema_version", "ticket_id", "decided_at", "masked_text", "pii_masked",
    "topic", "topic_confidence", "topic_margin", "risk_level", "risk_flags",
    "kb_top", "action", "rule_id", "reasons", "decided_by",
    "versions", "latency_ms", "answer_sent",
]

SCHEMA_VERSION = "1.0"


class AuditLog:
    def __init__(self, path: str = DEFAULT_LOG_PATH) -> None:
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(record)
        record["schema_version"] = SCHEMA_VERSION
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError("В аудит-записи отсутствуют поля: %s" % ", ".join(missing))
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def reset(self) -> None:
        """Очистить лог. Используется только демо-скриптом, чтобы каждый запуск
        начинался с чистого состояния и вывод был воспроизводим."""
        if os.path.exists(self.path):
            os.remove(self.path)
