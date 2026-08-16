"""Токенизация и TF-IDF на чистой стандартной библиотеке Python.

TF-IDF (Term Frequency - Inverse Document Frequency, частота слова, взвешенная
обратной частотой документа) - способ превратить текст в вектор чисел, чтобы
можно было измерять похожесть двух текстов.

УПРОЩЕНИЕ: словарный TF-IDF с грубым «стеммингом» обрезкой слова до 5 символов.
В целевой системе - многоязычная модель эмбеддингов multilingual-e5-small
(открытая модель, Apache-2.0) + векторная база (pgvector / Qdrant).
Причина упрощения: PoC должен запускаться без интернета и без установки
зависимостей, чтобы жюри могло воспроизвести результат за 10 секунд.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Tuple

# Слова, которые встречаются почти в каждом тикете и не несут смысла для темы.
STOPWORDS = {
    "это", "как", "что", "для", "или", "так", "уже", "все", "его", "мне", "мой",
    "моя", "моё", "вас", "вам", "меня", "быть", "если", "него", "чтобы", "потому",
    "здравствуйте", "добрый", "день", "пожалуйста", "спасибо", "подскажите",
    "скажите", "хочу", "нужно", "можно", "есть", "нет", "они", "она", "оно",
}

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")
_STEM_LEN = 5


def tokenize(text: str) -> List[str]:
    """Текст -> список нормализованных токенов.

    Шаги: нижний регистр -> выделение буквенно-цифровых слов -> отсев коротких
    слов и стоп-слов -> обрезка до 5 символов (грубая замена морфологии:
    «заказа», «заказу», «заказом» превращаются в один токен «заказ»).
    """
    tokens = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if len(raw) < 3 or raw in STOPWORDS:
            continue
        tokens.append(raw[:_STEM_LEN])
    return tokens


class TfidfIndex:
    """Индекс документов с поиском по косинусной близости.

    Косинусная близость cos(A, B) = (A · B) / (|A| * |B|), где
      A · B  - скалярное произведение векторов документа и запроса,
      |A|    - длина (норма) вектора A.
    Значение от 0 (нет общих слов) до 1 (векторы совпадают).
    """

    def __init__(self, documents: Iterable[str]) -> None:
        self.docs: List[str] = list(documents)
        self.doc_tokens: List[List[str]] = [tokenize(d) for d in self.docs]
        n_docs = max(len(self.docs), 1)

        # document frequency: в скольких документах встретился токен
        df: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        # IDF(t) = ln((N + 1) / (df(t) + 1)) + 1, где N - число документов.
        # Сглаживание +1 нужно, чтобы не делить на ноль для новых слов.
        self.idf: Dict[str, float] = {
            token: math.log((n_docs + 1) / (freq + 1)) + 1.0 for token, freq in df.items()
        }
        self.doc_vectors: List[Dict[str, float]] = [
            self._vectorize(tokens) for tokens in self.doc_tokens
        ]

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        counts: Dict[str, float] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0.0) + 1.0
        # TF(t) = число вхождений t / длину документа
        length = float(len(tokens))
        vector = {
            token: (count / length) * self.idf.get(token, 1.0)
            for token, count in counts.items()
        }
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {token: v / norm for token, v in vector.items()}

    def vectorize_query(self, text: str) -> Dict[str, float]:
        return self._vectorize(tokenize(text))

    def search(self, text: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Вернуть [(индекс документа, косинусная близость)], топ-K по убыванию."""
        query = self.vectorize_query(text)
        if not query:
            return []
        scored = []
        for idx, doc_vector in enumerate(self.doc_vectors):
            # Оба вектора уже нормированы, поэтому скалярное произведение
            # и есть косинусная близость.
            score = sum(weight * doc_vector.get(token, 0.0) for token, weight in query.items())
            if score > 0.0:
                scored.append((idx, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]
