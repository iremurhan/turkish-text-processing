"""Lemma-POS lexicon built from Universal Dependencies treebanks.

Used when a word arrives without a part-of-speech tag: the parser then tries every category the
lemma has in the lexicon.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..conllu import read_conllu

EXCLUDED = {"NUM", "X", "SYM"}


def build_lexicon(conllu_files: list[str | Path]) -> dict[str, set[str]]:
    lexicon: dict[str, set[str]] = defaultdict(set)
    for path in conllu_files:
        for sentence in read_conllu(path):
            for w in sentence.words:
                if w.upos not in EXCLUDED and w.lemma not in ("_", ""):
                    lexicon[w.lemma.lower()].add(w.upos)
    return dict(lexicon)


def save_lexicon(lexicon: dict[str, set[str]], path: str | Path) -> int:
    pairs = sorted((lemma, pos) for lemma, tags in lexicon.items() for pos in tags)
    with open(path, "w", encoding="utf-8") as f:
        for lemma, pos in pairs:
            f.write(f"{lemma}\t{pos}\n")
    return len(pairs)


def load_lexicon(path: str | Path) -> dict[str, set[str]]:
    lexicon: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            lemma, pos = line.rstrip("\n").split("\t")
            lexicon[lemma].add(pos)
    return dict(lexicon)
