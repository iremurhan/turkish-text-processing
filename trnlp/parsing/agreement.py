"""Agreement constraints checked on a parse tree using the words' morphological features.

A sentence is accepted if at least one of its parse trees satisfies every constraint:

1. subject-verb agreement in person and number (a third-person plural subject may take a singular verb);
2. determiner-noun agreement: "bir" cannot modify a plural noun;
3. time adverbs and tense: "dün" requires past tense, "yarın" excludes it;
4. verb frames: intransitive verbs take no bare (nominative) object, and some verbs exclude a case;
5. voice: a passive verb cannot have a first- or second-person pronoun as its subject;
6. possessives: a pronoun that modifies a noun must be in the genitive ("bizim köy", not "ben okul").
"""

from __future__ import annotations

from ..conllu import Word
from .cky import Tree

TIME_ADVERBS = {"dün": {"required": {"Past"}}, "yarın": {"excluded": {"Past"}}}

VERB_FRAMES = {
    "git": {"transitive": False, "excluded_cases": {"Loc"}},
    "gel": {"transitive": False, "excluded_cases": {"Loc"}},
    "koş": {"transitive": False, "excluded_cases": set()},
    "uyu": {"transitive": False, "excluded_cases": set()},
}

NOMINAL = {"NOUN", "PROPN", "PRON"}


def _head(tree: Tree, words: list[Word]) -> Word | None:
    """Turkish phrases are head-final: the head is the rightmost nominal word."""
    for i in reversed(tree.leaves()):
        if words[i].upos in NOMINAL:
            return words[i]
    return None


def _case(w: Word) -> str:
    return w.feats.get("Case", "Nom")


def _person(w: Word) -> str:
    return w.feats.get("Person", "3")


def _number(w: Word) -> str:
    return w.feats.get("Number", "Sing")


def _nodes(tree: Tree):
    yield tree
    for c in tree.children:
        yield from _nodes(c)


def _predicate(words: list[Word]) -> tuple[Word | None, Word | None]:
    """(finite verb or copula carrying the tense, word carrying person/number)."""
    verb = next((w for w in reversed(words) if w.upos in ("VERB", "AUX") and w.lemma.lower() not in
                 ("mi", "mı", "mu", "mü")), None)
    particle = next((w for w in reversed(words) if w.lemma.lower() in ("mi", "mı", "mu", "mü") and "Person" in w.feats), None)
    return verb, particle or verb


def _subject(tree: Tree, words: list[Word]) -> Word | None:
    """The nominative head of the NP directly under S1, if there is one (Turkish allows pro-drop)."""
    for node in _nodes(tree):
        if node.label == "S1" and node.children and node.children[0].label == "NP":
            head = _head(node.children[0], words)
            if head is not None and _case(head) == "Nom":
                return head
    return None


def violations(tree: Tree, words: list[Word]) -> list[str]:
    found = []
    verb, agr = _predicate(words)
    subject = _subject(tree, words)

    # 1. subject-verb agreement
    if subject is not None and agr is not None:
        sp, sn, vp, vn = _person(subject), _number(subject), _person(agr), _number(agr)
        if sp != vp:
            found.append(f"subject-verb person: {subject.form} ({sp}) vs {agr.form} ({vp})")
        elif (sp in ("1", "2") and sn != vn) or (sp == "3" and sn == "Sing" and vn == "Plur"):
            found.append(f"subject-verb number: {subject.form} ({sn}) vs {agr.form} ({vn})")

    for node in _nodes(tree):
        if len(node.children) != 2:
            continue
        left, right = node.children
        # 2. determiner-noun number
        if node.label == "NP" and left.label == "DET" and words[left.index].lemma.lower() == "bir":
            head = _head(right, words)
            if head is not None and _number(head) == "Plur":
                found.append(f"determiner-noun number: bir {head.form}")
        # 6. pronoun as a modifier must be genitive
        if node.label == "NP" and left.label == "NP" and right.label == "NP":
            head = _head(left, words)
            if head is not None and head.upos == "PRON" and _case(head) != "Gen":
                found.append(f"possessive: {head.form} is not genitive")

    if verb is not None:
        tense = verb.feats.get("Tense")
        # 3. time adverb and tense
        for w in words:
            rule = TIME_ADVERBS.get(w.lemma.lower())
            if rule and tense:
                if "required" in rule and tense not in rule["required"]:
                    found.append(f"tense: {w.form} with {tense}")
                if tense in rule.get("excluded", ()):
                    found.append(f"tense: {w.form} with {tense}")
        # 4. verb frame
        frame = VERB_FRAMES.get(verb.lemma.lower())
        if frame:
            for node in _nodes(tree):
                if node.label == "VP" and len(node.children) == 2 and node.children[0].label == "NP":
                    head = _head(node.children[0], words)
                    if head is None or head is subject:
                        continue
                    if not frame["transitive"] and _case(head) == "Nom":
                        found.append(f"verb frame: intransitive {verb.lemma} with bare object {head.form}")
                    if _case(head) in frame["excluded_cases"]:
                        found.append(f"verb frame: {verb.lemma} with {_case(head)} {head.form}")
        # 5. passive voice with a first/second-person subject
        if verb.feats.get("Voice") == "Pass" and subject is not None and subject.upos == "PRON" \
                and _person(subject) in ("1", "2"):
            found.append(f"voice: passive {verb.form} with subject {subject.form}")
    return sorted(set(found))


def parse_with_agreement(parser, words: list[Word]) -> tuple[list[Tree], list[Tree], list[list[str]]]:
    """(all trees, trees that satisfy every constraint, violations of the rejected trees)."""
    trees = parser.trees(words)
    valid, rejected = [], []
    for t in trees:
        v = violations(t, words)
        (rejected if v else valid).append(v if v else t)
    return trees, valid, rejected
