"""Классификатор темы тикета: правила + kNN на TF-IDF.

Два независимых сигнала складываются в один балл на категорию:
  1. Правила  - совпадения ключевых слов (высокая точность, низкая полнота).
  2. kNN      - k Nearest Neighbours, k ближайших соседей: ищем самые похожие
                размеченные примеры и голосуем их метками (ловит формулировки,
                которых нет в списке ключевых слов).

УПРОЩЕНИЕ: kNN по 40 размеченным примерам. В целевой системе - линейная модель
или дообученный rubert-tiny2 на ~20 000 размеченных исторических тикетов,
см. docs/ml.md.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

from .taxonomy import CATEGORIES, CLASSIFIER_VERSION, RULE_KEYWORDS, UNKNOWN_CATEGORY
from .textutil import TfidfIndex, tokenize

# Вес правил против вес kNN в итоговом балле. Сумма = 1.0.
W_RULES = 0.6
W_KNN = 0.4
KNN_K = 5


class TopicPrediction(NamedTuple):
    category: str          # предсказанная тема
    confidence: float      # балл топ-1 категории, 0..1
    margin: float          # отрыв топ-1 от топ-2, 0..1
    scores: Dict[str, float]
    version: str


class TopicClassifier:
    def __init__(self, examples: List[Dict[str, str]]) -> None:
        self.example_labels = [item["label"] for item in examples]
        self.index = TfidfIndex([item["text"] for item in examples])

    def _rule_scores(self, text: str) -> Dict[str, float]:
        tokens = set(tokenize(text))
        scores = {category: 0.0 for category in CATEGORIES}
        for category, keywords in RULE_KEYWORDS.items():
            hits = sum(1 for keyword in keywords if keyword in tokens)
            if hits:
                # Насыщение: 3 совпадения и больше дают максимум 1.0, чтобы одна
                # многословная категория не забивала остальные.
                scores[category] = min(hits / 3.0, 1.0)
        return scores

    def _knn_scores(self, text: str) -> Dict[str, float]:
        scores = {category: 0.0 for category in CATEGORIES}
        neighbours: List[Tuple[int, float]] = self.index.search(text, top_k=KNN_K)
        for idx, similarity in neighbours:
            scores[self.example_labels[idx]] += similarity
        return scores

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        total = sum(scores.values())
        if total <= 0.0:
            return {category: 0.0 for category in scores}
        return {category: value / total for category, value in scores.items()}

    def predict(self, text: str) -> TopicPrediction:
        rules = self._normalize(self._rule_scores(text))
        knn = self._normalize(self._knn_scores(text))
        combined = {
            category: W_RULES * rules[category] + W_KNN * knn[category]
            for category in CATEGORIES
        }
        ranked = sorted(combined.items(), key=lambda pair: (-pair[1], pair[0]))
        top_category, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # Ни одно правило не сработало и ни один размеченный пример не оказался
        # похож: все баллы нулевые. Сортировка в этом случае вернула бы просто
        # первую категорию по алфавиту (account_access) — и в аудит-лог попала
        # бы тема, которую система на самом деле не определяла.
        if top_score <= 0.0:
            return TopicPrediction(
                category=UNKNOWN_CATEGORY,
                confidence=0.0,
                margin=0.0,
                scores={category: 0.0 for category in CATEGORIES},
                version=CLASSIFIER_VERSION,
            )

        return TopicPrediction(
            category=top_category,
            confidence=round(top_score, 4),
            margin=round(top_score - second_score, 4),
            scores={category: round(value, 4) for category, value in combined.items()},
            version=CLASSIFIER_VERSION,
        )
