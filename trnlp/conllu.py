"""Minimal CoNLL-U reader that keeps both surface tokens (with spacing) and syntactic words."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    form: str
    lemma: str
    upos: str
    feats: dict[str, str]


@dataclass
class Sentence:
    tokens: list[str] = field(default_factory=list)       # surface tokens; multiword tokens kept whole
    space_after: list[bool] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)       # syntactic words; multiword tokens split

    @property
    def text(self) -> str:
        """Sentence text rebuilt from the surface tokens and SpaceAfter information."""
        return "".join(t + (" " if s else "") for t, s in zip(self.tokens, self.space_after)).rstrip()


def parse_feats(value: str) -> dict[str, str]:
    if value in ("_", ""):
        return {}
    return dict(item.split("=", 1) for item in value.split("|") if "=" in item)


def _space_after(misc: str) -> bool:
    return "SpaceAfter=No" not in misc


def read_conllu(path: str | Path) -> list[Sentence]:
    sentences, current, skip_until = [], Sentence(), 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current.words:
                    sentences.append(current)
                current, skip_until = Sentence(), 0
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 10:
                cols += ["_"] * (10 - len(cols))
            idx = cols[0]
            if "." in idx:  # empty nodes
                continue
            if "-" in idx:  # multiword token: its parts are words, the range is the surface token
                start, end = (int(x) for x in idx.split("-"))
                current.tokens.append(cols[1])
                current.space_after.append(_space_after(cols[9]))
                skip_until = end
                continue
            current.words.append(Word(cols[1], cols[2], cols[3], parse_feats(cols[5])))
            if int(idx) > skip_until:
                current.tokens.append(cols[1])
                current.space_after.append(_space_after(cols[9]))
    if current.words:
        sentences.append(current)
    return sentences
