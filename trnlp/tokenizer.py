"""Tokenization for Turkish: a rule-based tokenizer and a Naive Bayes boundary classifier."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .conllu import Sentence
from .patterns import DEFAULT_MWES, special_pattern

# A word keeps apostrophe suffixes (Ankara'da, 2023'ün) and inner hyphens (e-posta) together.
_WORD = r"\w+(?:[-'’]\w+)*"


_LOWER = "a-zçğıöşü"
# '5.' before a lower-case word is an ordinal ("5. sergi"), kept as one token.
_ORDINAL = rf"\b\d+\.(?=\s+[{_LOWER}])"


def _abbreviation_pattern() -> str:
    from .resources import abbreviations

    dotted = sorted((a for a in abbreviations() if a.endswith(".")), key=len, reverse=True)
    return r"(?<!\w)(?:" + "|".join(re.escape(a) for a in dotted) + r")"


class RuleTokenizer:
    """Regex tokenizer. Order of preference: special entities and multi-word expressions,
    dotted abbreviations (TDK list) and ordinals, words, then single symbols."""

    def __init__(self, mwes=DEFAULT_MWES):
        special = f"(?:{special_pattern(mwes).pattern})(?:['’]\\w+)?"  # with suffix: 3.12.1999'da, Kız Kulesi'ni
        parts = [_ORDINAL, special, _abbreviation_pattern(), _WORD, r"\.\.\.|…", r"[^\w\s]"]
        self.pattern = re.compile("|".join(f"(?:{p})" for p in parts), re.IGNORECASE)

    def __call__(self, text: str) -> list[str]:
        return [m.group(0) for m in self.pattern.finditer(text)]

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in self.pattern.finditer(text)]


# ---------------------------------------------------------------------------
# Naive Bayes boundary classifier
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    "base": ("c", "l", "r"),                         # the character and its neighbours
    "case": ("c", "l", "r", "lcap", "rcap"),         # + capitalisation of the neighbours
    "wide": ("c", "ll", "l", "r", "rr"),             # + two characters on each side
    "full": ("c", "ll", "l", "r", "rr", "lcap", "rcap", "lt", "rt", "rsp"),
}


def _ctype(ch: str) -> str:
    if not ch:
        return "none"
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        return "upper" if ch.isupper() else "lower"
    if ch.isspace():
        return "space"
    return "punct"


def is_candidate(ch: str) -> bool:
    """Whitespace and punctuation are the only places where a token boundary can be."""
    return ch.isspace() or not ch.isalnum()


def candidate_features(text: str, i: int, names) -> list[str]:
    left = text[:i].rstrip()
    right = text[i + 1 :].lstrip()
    l, ll = left[-1:] or "", left[-2:-1] or ""
    r, rr = right[:1] or "", right[1:2] or ""
    values = {
        "c": "␣" if text[i].isspace() else text[i],
        "l": l, "ll": ll, "r": r, "rr": rr,
        "lcap": str(l.isupper()), "rcap": str(r.isupper()),
        "lt": _ctype(l), "rt": _ctype(r),
        "rsp": str(i + 1 < len(text) and text[i + 1].isspace()),
    }
    return [f"{n}={values[n]}" for n in names]


def boundary_examples(sentence: Sentence):
    """(text, position, label) for every candidate character of a treebank sentence.

    Label is True when the character separates tokens: whitespace between tokens, or a
    punctuation character that is (part of) a token made only of punctuation. Punctuation
    inside a word or number ("3.12.1999", "Ankara'da", "e-posta") is labelled False.
    """
    text, pos, owner = "", 0, []
    for token, space in zip(sentence.tokens, sentence.space_after):
        punct_only = not any(ch.isalnum() for ch in token)
        for _ in token:
            owner.append(punct_only)
        text += token
        if space:
            text += " "
            owner.append(None)  # whitespace outside tokens
    text = text.rstrip()
    owner = owner[: len(text)]
    for i, ch in enumerate(text):
        if not is_candidate(ch):
            continue
        label = True if owner[i] is None else bool(owner[i])
        yield text, i, label


class NaiveBayesTokenizer:
    """Multinomial Naive Bayes over categorical context features, with add-alpha smoothing."""

    def __init__(self, features: str = "full", alpha: float = 1.0):
        self.names = FEATURE_SETS[features]
        self.alpha = alpha

    def fit(self, sentences: list[Sentence]) -> "NaiveBayesTokenizer":
        class_counts = Counter()
        feat_counts = {True: Counter(), False: Counter()}
        values = defaultdict(set)
        for s in sentences:
            for text, i, label in boundary_examples(s):
                class_counts[label] += 1
                for f in candidate_features(text, i, self.names):
                    feat_counts[label][f] += 1
                    values[f.split("=", 1)[0]].add(f)
        total = sum(class_counts.values())
        self.prior = {c: math.log(class_counts[c] / total) for c in (True, False)}
        self.vocab = {name: len(v) for name, v in values.items()}
        self.counts = feat_counts
        self.class_totals = {c: class_counts[c] for c in (True, False)}
        return self

    def _log_likelihood(self, feature: str, label: bool) -> float:
        name = feature.split("=", 1)[0]
        count = self.counts[label][feature]
        return math.log((count + self.alpha) / (self.class_totals[label] + self.alpha * (self.vocab.get(name, 0) + 1)))

    def predict_one(self, text: str, i: int) -> bool:
        feats = candidate_features(text, i, self.names)
        scores = {c: self.prior[c] + sum(self._log_likelihood(f, c) for f in feats) for c in (True, False)}
        return scores[True] > scores[False]

    def __call__(self, text: str) -> list[str]:
        tokens, current = [], ""
        for i, ch in enumerate(text):
            if is_candidate(ch) and self.predict_one(text, i):
                if current:
                    tokens.append(current)
                current = ""
                if not ch.isspace():
                    if tokens and not any(c.isalnum() for c in tokens[-1]) and text[i - 1] == tokens[-1][-1] == ch:
                        tokens[-1] += ch  # repeated symbol such as "..." stays one token
                    else:
                        tokens.append(ch)
            else:
                current += ch
        if current:
            tokens.append(current)
        return tokens


def evaluate_boundaries(model: NaiveBayesTokenizer, sentences: list[Sentence], punctuation_only: bool = True) -> dict:
    """Precision/recall/F1 for the 'splits here' class and accuracy over candidate characters."""
    tp = fp = fn = tn = 0
    for s in sentences:
        for text, i, gold in boundary_examples(s):
            if punctuation_only and text[i].isspace():
                continue
            pred = model.predict_one(text, i)
            tp += pred and gold
            fp += pred and not gold
            fn += gold and not pred
            tn += not pred and not gold
    return _prf(tp, fp, fn, tn)


def evaluate_tokens(tokenize, sentences: list[Sentence]) -> dict:
    """Token-level precision/recall/F1: a predicted token counts if its character span matches a gold token."""
    tp = n_pred = n_gold = 0
    for s in sentences:
        text, gold = "", set()
        for token, space in zip(s.tokens, s.space_after):
            gold.add((len(text), len(text) + len(token)))
            text += token + (" " if space else "")
        text = text.rstrip()
        pred, start = set(), 0
        for token in tokenize(text):
            begin = text.find(token, start)
            if begin < 0:
                continue
            pred.add((begin, begin + len(token)))
            start = begin + len(token)
        tp += len(gold & pred)
        n_pred += len(pred)
        n_gold += len(gold)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _prf(tp, fp, fn, tn) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "accuracy": (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
