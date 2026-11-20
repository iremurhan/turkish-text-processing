"""PARSEVAL-style labelled bracket scores and a reader for bracketed trees."""

from __future__ import annotations

import re


def bracket_scores(predicted: set, gold: set) -> dict[str, float]:
    """Labelled precision, recall and F1 over (label, start, end) constituents."""
    correct = len(predicted & gold)
    p = correct / len(predicted) if predicted else 0.0
    r = correct / len(gold) if gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def read_bracketed(text: str) -> set[tuple[str, int, int]]:
    """Constituents of a tree written as '[S [S1 [ADV Dün] ...] [PUNCT .]]', 1-based word positions.

    Pre-terminals (a label followed directly by a word) are not counted, matching `Tree.constituents`.
    """
    tokens = re.findall(r"\[|\]|[^\s\[\]]+", text)
    stack, spans, position, i = [], set(), 0, 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "[":
            label = tokens[i + 1]
            if i + 3 < len(tokens) and tokens[i + 2] not in "[]" and tokens[i + 3] == "]":
                position += 1  # pre-terminal: [LABEL word]
                i += 4
                continue
            stack.append((label, position + 1))
            i += 2
            continue
        if tok == "]":
            label, start = stack.pop()
            spans.add((label, start, position))
        i += 1
    return spans
