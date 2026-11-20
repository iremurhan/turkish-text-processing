"""Context-free grammar (in Chomsky normal form) for simple SOV Turkish sentences.

The parser expects morphologically analysed input (CoNLL-U style: lemma, UPOS, features), as if a
morphological analyser and disambiguator had run before it. Each word is first mapped to a
pre-terminal category; the binary rules below then combine categories.
"""

from __future__ import annotations

# Binary rules A -> B C, designed from the declarative, interrogative and imperative example sentences.
RULES: list[tuple[str, str, str]] = [
    # sentence
    ("S", "S1", "PUNCT"),
    ("S1", "NP", "VP"),
    ("S1", "ADV", "VP"),
    ("S1", "ADP", "VP"),
    # noun phrases
    ("NP", "DET", "NP"),
    ("NP", "ADJ", "NP"),
    ("NP", "NP", "NP"),     # compounds and possessives: "akşam yemeği", "bizim köye"
    ("NP", "X1", "NP"),     # coordination: "kültürümüzü ve tarihimizi"
    ("X1", "NP", "CONJ"),
    # adpositional phrases
    ("ADP", "NP", "ADP"),   # "akşam yemeği için", "ağacın altında"
    ("ADP", "DET", "ADP"),
    ("ADP", "ADJ", "ADP"),
    # adjectives and adverbs
    ("ADJ", "ADV", "ADJ"),  # "en güzel"
    ("ADJ", "NP", "PCP"),   # "akşam -ki"
    ("ADV", "ADJ", "NP"),   # "yüksek sesle", "en son ne zaman"
    ("ADV", "ADJ", "ADV"),
    # verb phrases
    ("VP", "NP", "VP"),
    ("VP", "ADV", "VP"),
    ("VP", "ADP", "VP"),
    ("VP", "NP", "COP"),    # nominal predicate: "meyve -dir"
    ("VP", "ADJ", "COP"),   # "uzakta -ydı"
    ("VP", "VP", "QSN"),    # yes/no question: "katılacak mısınız"
]

QUESTION_PARTICLES = {"mi", "mı", "mu", "mü"}

UPOS_TO_CATEGORY = {
    "NOUN": "NP", "PROPN": "NP", "PRON": "NP",
    "VERB": "VP",
    "ADJ": "ADJ", "ADV": "ADV", "DET": "DET", "ADP": "ADP",
    "CCONJ": "CONJ", "PUNCT": "PUNCT",
}


def category(lemma: str, upos: str) -> str | None:
    """Pre-terminal category of an analysed word."""
    lemma = lemma.lower()
    if lemma in QUESTION_PARTICLES:
        return "QSN"
    if lemma == "ki":
        return "PCP"
    if upos == "AUX":
        return "COP"
    return UPOS_TO_CATEGORY.get(upos)


def binary_index(rules=RULES) -> dict[tuple[str, str], list[str]]:
    index: dict[tuple[str, str], list[str]] = {}
    for lhs, left, right in rules:
        index.setdefault((left, right), []).append(lhs)
    return index
