"""Тесты PoC. Только стандартная библиотека: python3 -m unittest discover -s tests

Тесты покрывают инварианты, нарушение которых означает небезопасную систему:
запрет автозакрытия денежных категорий, карантин prompt injection, маскирование
персональных данных, корректная деградация и полнота аудит-записи.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poc.audit import REQUIRED_FIELDS, AuditLog
from poc.classifier import TopicClassifier
from poc.llm import CircuitBreaker, LlmUnavailable, build_provider, generate_with_breaker
from poc.pipeline import Pipeline
from poc.retrieval import KnowledgeBase, TicketHistory
from poc.risk import mask_pii
from poc.taxonomy import HOT_PATH_SLO_MS, NEVER_AUTO_CLOSE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_jsonl(name):
    rows = []
    with open(os.path.join(ROOT, "data", name), "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_pipeline(provider_mode="mock", log_path=None):
    tickets = load_jsonl("tickets.jsonl")
    kb = KnowledgeBase(load_jsonl("kb.jsonl"))
    classifier = TopicClassifier(load_jsonl("labeled_examples.jsonl"))
    history = TicketHistory(tickets)
    audit = AuditLog(log_path or os.path.join(tempfile.mkdtemp(), "decisions.jsonl"))
    pipeline = Pipeline(classifier, kb, history, build_provider(provider_mode), audit)
    return pipeline, {t["id"]: t for t in tickets}


class TestPoc(unittest.TestCase):
    def setUp(self):
        self.pipeline, self.tickets = build_pipeline()

    def test_happy_path_gives_auto_answer(self):
        """Простой вопрос о доставке закрывается автоматически со ссылкой на статью."""
        result = self.pipeline.process(self.tickets["T-1001"])
        self.assertEqual(result["effective_action"], "auto_answer")
        self.assertTrue(result["audit"]["answer_sent"])
        self.assertTrue(result["audit"]["kb_top"], "автоответ обязан опираться на статью базы знаний")

    def test_prompt_injection_is_quarantined(self):
        """Тикет с попыткой перехвата инструкций не доходит до генерации."""
        result = self.pipeline.process(self.tickets["T-1010"])
        self.assertEqual(result["effective_action"], "escalate_operator")
        self.assertFalse(result["audit"]["answer_sent"])
        self.assertIn("prompt_injection", result["audit"]["risk_flags"])
        self.assertEqual(result["audit"]["draft_source"], "none",
                         "при injection LLM не должен вызываться вообще")

    def test_restricted_categories_are_never_auto_closed(self):
        """Ни один тикет из денежных/защищённых категорий не закрывается сам."""
        for ticket in self.tickets.values():
            result = self.pipeline.process(ticket)
            if result["audit"]["topic"] in NEVER_AUTO_CLOSE:
                self.assertNotEqual(result["effective_action"], "auto_answer",
                                    "тикет %s закрыт автоматически" % ticket["id"])
                self.assertFalse(result["audit"]["answer_sent"])

    def test_low_confidence_goes_to_triage(self):
        """Непонятный тикет уходит человеку, а не угадывается."""
        result = self.pipeline.process(self.tickets["T-1014"])
        self.assertEqual(result["effective_action"], "route_to_queue")
        self.assertEqual(result["audit"]["queue"], "general_triage")

    def test_pii_is_masked(self):
        """Телефон, почта и номер карты не попадают дальше приёма."""
        masked = mask_pii("Телефон +7 916 123-45-67, почта a.b@example.com, карта 4276 3800 1234 5678")
        self.assertNotIn("916", masked["text"])
        self.assertNotIn("example.com", masked["text"])
        self.assertNotIn("4276", masked["text"])
        self.assertEqual(set(masked["pii_found"]), {"PHONE", "EMAIL", "CARD"})

    def test_audit_record_is_complete(self):
        """В аудит-записи есть все поля, нужные для воспроизведения решения."""
        result = self.pipeline.process(self.tickets["T-1004"])
        for field in REQUIRED_FIELDS:
            self.assertIn(field, result["audit"])
        self.assertIn("classifier", result["audit"]["versions"])
        with open(self.pipeline.audit.path, "r", encoding="utf-8") as handle:
            for line in handle:
                json.loads(line)  # каждая строка обязана быть валидным JSON

    def test_hot_path_within_slo(self):
        """Горячий путь укладывается в бюджет для всех тикетов датасета."""
        for ticket in self.tickets.values():
            hot = self.pipeline.run_hot_path(ticket)
            self.assertLessEqual(hot["hot_path_ms"], HOT_PATH_SLO_MS)


class TestFallback(unittest.TestCase):
    def test_llm_outage_degrades_without_auto_close(self):
        """При недоступности LLM автоответ отменяется, SLA горячего пути цел."""
        pipeline, tickets = build_pipeline(provider_mode="failing")
        result = pipeline.process(tickets["T-1013"])
        self.assertTrue(result["cold"]["degraded"])
        self.assertEqual(result["cold"]["draft_source"], "template-fallback")
        self.assertFalse(result["audit"]["answer_sent"], "в деградации автоответ запрещён")
        self.assertEqual(result["effective_action"], "draft_for_operator")
        self.assertTrue(result["hot"]["sla_ok"])

    def test_circuit_breaker_opens_and_short_circuits(self):
        """После 3 подряд отказов предохранитель размыкается и экономит таймауты."""
        breaker = CircuitBreaker(failure_threshold=3, cooldown_s=30.0)
        provider = build_provider("failing")
        first = generate_with_breaker(provider, breaker, "текст", "заголовок", "статья")
        self.assertFalse(first.ok)
        self.assertEqual(breaker.state, "open")
        second = generate_with_breaker(provider, breaker, "текст", "заголовок", "статья")
        self.assertIn("circuit_breaker_open", second.error)

    def test_failing_provider_raises_expected_exception(self):
        with self.assertRaises(LlmUnavailable):
            build_provider("failing").generate("t", "k", "v")


if __name__ == "__main__":
    unittest.main(verbosity=2)
