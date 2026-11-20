"""Sentence splitting: a rule-based splitter and a logistic-regression boundary classifier.

Both work on tokens. A *candidate* is a token that ends in '.', '!', '?' or '…'; the task is to
decide whether the sentence ends there. Closing quotes and brackets right after the candidate
stay with the sentence they close.
"""

from __future__ import annotations

import re

import numpy as np

from .conllu import Sentence
from .resources import dotted_abbreviations
from .tokenizer import RuleTokenizer

ENDERS = (".", "!", "?", "…")
CLOSERS = {'"', "”", "’", "'", ")", "]", "»"}
_NUMBER = re.compile(r"^\d+([.,]\d+)?\.?$")


def is_candidate(token: str) -> bool:
    return token.endswith(ENDERS)


def _starts_lower(token: str) -> bool:
    return bool(token) and token[0].isalpha() and token[0].islower()


def _is_abbreviation(token: str, prev: str) -> bool:
    abbr = dotted_abbreviations()
    if token == ".":
        return (prev.lower() + ".") in abbr
    return token.lower() in abbr


def _is_ordinal(token: str, prev: str, nxt: str) -> bool:
    """'5.' or '5 .' followed by a lower-case word is an ordinal number, not a sentence end."""
    number = token[:-1] if token != "." else prev
    return number.isdigit() and _starts_lower(nxt)


class RuleSplitter:
    """End a sentence at '.', '!', '?' or '…' unless it closes an abbreviation, an ordinal or a decimal."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or RuleTokenizer()

    def is_boundary(self, tokens: list[str], i: int) -> bool:
        token = tokens[i]
        prev = tokens[i - 1] if i > 0 else ""
        nxt = _next_word(tokens, i)
        if token.endswith(".") and (_is_abbreviation(token, prev) or _is_ordinal(token, prev, nxt)):
            return False
        return True

    def split_tokens(self, tokens: list[str]) -> list[list[str]]:
        return _split_by(tokens, lambda i: self.is_boundary(tokens, i))

    def __call__(self, text: str) -> list[str]:
        return [" ".join(s) for s in self.split_tokens(self.tokenizer(text))]


FEATURES = {
    # names follow the 2023 report: token length, next-token capitalisation, numeric status, ...
    "small": ("tl", "ntc", "ns"),
    "medium": ("tl", "ntc", "ns", "ptl", "ntl"),
    "full": ("tl", "ntc", "ns", "ptl", "ntl", "abbr", "pnum", "ncl", "dot", "nlow", "nnum", "end"),
}


def token_features(tokens: list[str], i: int, names) -> list[float]:
    token = tokens[i]
    prev = tokens[i - 1] if i > 0 else ""
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
    word = _next_word(tokens, i)
    values = {
        "tl": len(token),
        "ntc": float(word[:1].isupper()),
        "ns": float(bool(_NUMBER.match(token))),
        "ptl": len(prev),
        "ntl": len(word),
        "abbr": float(_is_abbreviation(token, prev)),
        "pnum": float(prev.isdigit()),
        "ncl": float(nxt in CLOSERS),
        "dot": float(token.endswith(".")),
        "nlow": float(_starts_lower(word)),
        "nnum": float(word[:1].isdigit()),
        "end": float(word == ""),
    }
    return [values[n] for n in names]


def _next_word(tokens: list[str], i: int) -> str:
    j = i + 1
    while j < len(tokens) and tokens[j] in CLOSERS:
        j += 1
    return tokens[j] if j < len(tokens) else ""


def _split_by(tokens: list[str], boundary) -> list[list[str]]:
    sentences, current, i = [], [], 0
    while i < len(tokens):
        current.append(tokens[i])
        if is_candidate(tokens[i]) and boundary(i):
            while i + 1 < len(tokens) and tokens[i + 1] in CLOSERS:
                i += 1
                current.append(tokens[i])
            sentences.append(current)
            current = []
        i += 1
    if current:
        sentences.append(current)
    return sentences


def treebank_stream(sentences: list[Sentence]) -> tuple[list[str], list[int]]:
    """Concatenate treebank sentences; return tokens and the candidate indices that end a sentence."""
    tokens, ends = [], set()
    for s in sentences:
        toks = s.tokens
        last = len(toks) - 1
        while last > 0 and toks[last] in CLOSERS:
            last -= 1
        tokens.extend(toks)
        ends.add(len(tokens) - len(toks) + last)
    return tokens, sorted(ends)


def candidate_dataset(tokens: list[str], ends, names):
    ends = set(ends)
    idx = [i for i, t in enumerate(tokens) if is_candidate(t)]
    X = np.array([token_features(tokens, i, names) for i in idx], dtype=float)
    y = np.array([int(i in ends) for i in idx])
    return idx, X, y


class LogRegSplitter:
    """Logistic regression over hand-designed token features (scikit-learn)."""

    def __init__(self, features: str = "full", tokenizer=None):
        self.names = FEATURES[features]
        self.tokenizer = tokenizer or RuleTokenizer()

    def fit(self, sentences: list[Sentence]) -> "LogRegSplitter":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        tokens, ends = treebank_stream(sentences)
        _, X, y = candidate_dataset(tokens, ends, self.names)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
        return self

    def is_boundary(self, tokens: list[str], i: int) -> bool:
        return bool(self.model.predict(np.array([token_features(tokens, i, self.names)]))[0])

    def split_tokens(self, tokens: list[str]) -> list[list[str]]:
        return _split_by(tokens, lambda i: self.is_boundary(tokens, i))

    def __call__(self, text: str) -> list[str]:
        return [" ".join(s) for s in self.split_tokens(self.tokenizer(text))]


def evaluate(splitter, sentences: list[Sentence]) -> dict:
    """Precision/recall/F1 of sentence-end decisions over candidate tokens of a treebank."""
    tokens, ends = treebank_stream(sentences)
    ends = set(ends)
    tp = fp = fn = tn = 0
    for i, t in enumerate(tokens):
        if not is_candidate(t):
            continue
        pred, gold = splitter.is_boundary(tokens, i), i in ends
        tp += pred and gold
        fp += pred and not gold
        fn += gold and not pred
        tn += not pred and not gold
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "accuracy": (tp + tn) / (tp + fp + fn + tn), "candidates": tp + fp + fn + tn}
