"""CKY chart parser that returns every parse tree licensed by the grammar."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..conllu import Word
from .grammar import RULES, binary_index, category


@dataclass(frozen=True)
class Tree:
    label: str
    children: tuple["Tree", ...] = ()
    index: int | None = None  # position of the word for a pre-terminal

    @property
    def is_leaf(self) -> bool:
        return self.index is not None

    def leaves(self) -> list[int]:
        return [self.index] if self.is_leaf else [i for c in self.children for i in c.leaves()]

    def to_string(self, words: list[Word]) -> str:
        if self.is_leaf:
            return f"[{self.label} {words[self.index].form}]"
        return f"[{self.label} " + " ".join(c.to_string(words) for c in self.children) + "]"

    def constituents(self) -> set[tuple[str, int, int]]:
        """(label, first word, last word) for every phrase above the pre-terminals, 1-based."""
        if self.is_leaf:
            return set()
        leaves = self.leaves()
        spans = {(self.label, leaves[0] + 1, leaves[-1] + 1)}
        for c in self.children:
            spans |= c.constituents()
        return spans


@dataclass
class CKYParser:
    rules: list = field(default_factory=lambda: list(RULES))
    start: str = "S"
    max_trees: int = 10_000
    lexicon: dict[str, set[str]] | None = None  # lemma -> UPOS tags, used for words without a tag

    def __post_init__(self):
        self.index = binary_index(self.rules)

    def categories(self, w: Word) -> set[str]:
        tags = {w.upos} if w.upos not in ("_", "") else (self.lexicon or {}).get(w.lemma.lower(), set())
        return {c for c in (category(w.lemma, t) for t in tags) if c is not None}

    def chart(self, words: list[Word]) -> list[list[dict[str, list]]]:
        n = len(words)
        table: list[list[dict[str, list]]] = [[{} for _ in range(n)] for _ in range(n)]
        for i, w in enumerate(words):
            for cat in self.categories(w):
                table[i][i][cat] = [None]
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                cell = table[i][j]
                for k in range(i, j):
                    for left in table[i][k]:
                        for right in table[k + 1][j]:
                            for lhs in self.index.get((left, right), ()):
                                cell.setdefault(lhs, []).append((k, left, right))
        return table

    def trees(self, words: list[Word]) -> list[Tree]:
        """All parse trees rooted in the start symbol (up to `max_trees`)."""
        if not words:
            return []
        table = self.chart(words)
        memo: dict[tuple[int, int, str], list[Tree]] = {}

        def build(i: int, j: int, label: str) -> list[Tree]:
            key = (i, j, label)
            if key in memo:
                return memo[key]
            out: list[Tree] = []
            for back in table[i][j].get(label, []):
                if back is None:
                    out.append(Tree(label, index=i))
                    continue
                k, left, right = back
                for lt in build(i, k, left):
                    for rt in build(k + 1, j, right):
                        out.append(Tree(label, (lt, rt)))
                        if len(out) >= self.max_trees:
                            break
            memo[key] = out
            return out

        return build(0, len(words) - 1, self.start)
