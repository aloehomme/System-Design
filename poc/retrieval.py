"""Поиск релевантной статьи базы знаний и похожих прошлых тикетов.

RAG (Retrieval-Augmented Generation - генерация с опорой на найденные
документы): ответ строится ТОЛЬКО по найденному фрагменту базы знаний, а не по
«памяти» модели. Это ключевая мера против выдумывания фактов: если ничего не
нашли - автоответ не отправляем.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple

from .textutil import TfidfIndex


class KbHit(NamedTuple):
    doc_id: str
    title: str
    text: str
    category: str
    auto_answer_allowed: bool
    score: float


class KnowledgeBase:
    def __init__(self, articles: List[Dict[str, object]]) -> None:
        self.articles = articles
        # Индексируем заголовок + тело: заголовок несёт много сигнала на коротких
        # запросах, тело - на длинных.
        self.index = TfidfIndex(
            ["%s %s" % (a["title"], a["text"]) for a in articles]
        )

    def search(self, text: str, top_k: int = 3) -> List[KbHit]:
        hits = []
        for idx, score in self.index.search(text, top_k=top_k):
            article = self.articles[idx]
            hits.append(
                KbHit(
                    doc_id=str(article["id"]),
                    title=str(article["title"]),
                    text=str(article["text"]),
                    category=str(article["category"]),
                    auto_answer_allowed=bool(article["auto_answer_allowed"]),
                    score=round(score, 4),
                )
            )
        return hits


class SimilarTicket(NamedTuple):
    ticket_id: str
    text: str
    score: float


class TicketHistory:
    """Поиск похожих прошлых тикетов: используется для дедупликации инцидентов
    и как контекст оператору. В целевой системе - тот же векторный индекс,
    что и база знаний, но с отдельным namespace."""

    def __init__(self, tickets: List[Dict[str, object]]) -> None:
        self.tickets = tickets
        self.index = TfidfIndex([str(t["text"]) for t in tickets])

    def search(self, text: str, exclude_id: str = "", top_k: int = 3) -> List[SimilarTicket]:
        results = []
        for idx, score in self.index.search(text, top_k=top_k + 1):
            ticket = self.tickets[idx]
            if str(ticket["id"]) == exclude_id:
                continue
            results.append(
                SimilarTicket(
                    ticket_id=str(ticket["id"]),
                    text=str(ticket["text"])[:80],
                    score=round(score, 4),
                )
            )
        return results[:top_k]
