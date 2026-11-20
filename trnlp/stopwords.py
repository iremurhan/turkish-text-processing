"""Stopword elimination with a static Turkish list or a list derived from corpus frequencies."""

from __future__ import annotations

from collections import Counter

from .normalizer import turkish_lower
from .resources import stopwords as static_stopwords


def dynamic_stopwords(tokens: list[str], top_fraction: float = 0.001, min_count: int = 1) -> set[str]:
    """The most frequent word types of a corpus: the top `top_fraction` of the vocabulary by frequency."""
    counts = Counter(turkish_lower(t) for t in tokens if any(ch.isalpha() for ch in t))
    k = max(1, round(len(counts) * top_fraction))
    return {w for w, c in counts.most_common(k) if c >= min_count}


def remove_stopwords(tokens: list[str], stopwords=None) -> list[str]:
    stop = static_stopwords() if stopwords is None else stopwords
    return [t for t in tokens if turkish_lower(t) not in stop]
