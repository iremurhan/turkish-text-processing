"""Bundled language resources."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=None)
def abbreviations() -> dict[str, str]:
    """Abbreviation -> expansion, extracted from the TDK abbreviation index (Kısaltmalar Dizini)."""
    with resources.files("trnlp").joinpath("data/abbreviations.json").open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def dotted_abbreviations() -> frozenset[str]:
    """Lower-cased abbreviations that end in a full stop (e.g. 'dr.', 'vb.'), used for sentence splitting."""
    return frozenset(a.lower() for a in abbreviations() if a.endswith("."))


@lru_cache(maxsize=None)
def stopwords() -> frozenset[str]:
    with resources.files("trnlp").joinpath("data/stopwords.txt").open(encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip())
