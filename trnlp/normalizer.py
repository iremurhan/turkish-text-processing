"""Normalization: Turkish-aware lower-casing, abbreviation expansion, punctuation removal.

Special entities (URLs, e-mails, hashtags, dates, numbers, currency, phone numbers) are kept as they are.
"""

from __future__ import annotations

import re

from .patterns import special_pattern
from .resources import abbreviations
from .tokenizer import RuleTokenizer

_UPPER_TO_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_AMBIGUOUS = re.compile(r"\b1\.\s")  # TDK lists several senses as "1. ... 2. ..."


def turkish_lower(text: str) -> str:
    """Lower-case with Turkish dotted/dotless i: 'IŞIK İzmir' -> 'ışık izmir'."""
    return text.translate(_UPPER_TO_LOWER).lower()


class Normalizer:
    def __init__(self, expand_abbreviations: bool = True, lowercase: bool = True, drop_punctuation: bool = True,
                 tokenizer=None):
        self.tokenizer = tokenizer or RuleTokenizer()
        self.special = re.compile(f"^(?:{special_pattern().pattern})$", re.IGNORECASE)
        self.expand = expand_abbreviations
        self.lowercase = lowercase
        self.drop_punctuation = drop_punctuation
        self.table = {k: v for k, v in abbreviations().items() if not _AMBIGUOUS.search(v)}

    def expand_token(self, token: str) -> str:
        for key in (token, token.rstrip("."), token + "."):
            if key in self.table:
                return self.table[key]
        return token

    def normalize_tokens(self, tokens: list[str]) -> list[str]:
        out = []
        for token in tokens:
            if self.special.match(token):
                out.append(token)
                continue
            looks_abbreviated = len(token) > 1 and (token.isupper() or token.endswith("."))
            if self.expand and looks_abbreviated:
                token = self.expand_token(token)
            if self.drop_punctuation and not any(ch.isalnum() for ch in token):
                continue
            out.append(turkish_lower(token) if self.lowercase else token)
        return out

    def __call__(self, text: str) -> str:
        return " ".join(self.normalize_tokens(self.tokenizer(text)))
