"""Command-line interface: python -m trnlp <command> ..."""

from __future__ import annotations

import argparse
import json
import sys


def _read_text(args) -> str:
    return open(args.file, encoding="utf-8").read() if args.file else sys.stdin.read()


def cmd_tokenize(args):
    from .tokenizer import RuleTokenizer

    print("\n".join(RuleTokenizer()(_read_text(args))))


def cmd_split(args):
    from .splitter import RuleSplitter

    print("\n".join(RuleSplitter()(_read_text(args))))


def cmd_normalize(args):
    from .normalizer import Normalizer

    print(Normalizer()(_read_text(args)))


def cmd_stopwords(args):
    from .stopwords import dynamic_stopwords, remove_stopwords
    from .tokenizer import RuleTokenizer

    tokens = RuleTokenizer()(_read_text(args))
    stop = dynamic_stopwords(tokens, args.top_fraction) if args.dynamic else None
    print(" ".join(remove_stopwords(tokens, stop)))


def cmd_evaluate(args):
    from . import splitter as sp
    from .conllu import read_conllu
    from .tokenizer import FEATURE_SETS, NaiveBayesTokenizer, RuleTokenizer, evaluate_boundaries, evaluate_tokens

    train, test = read_conllu(args.train), read_conllu(args.test)
    rnd = lambda d: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
    results = {"tokenizer": {"rule": rnd(evaluate_tokens(RuleTokenizer(), test))}}
    for name in FEATURE_SETS:
        nb = NaiveBayesTokenizer(name).fit(train)
        results["tokenizer"][f"naive_bayes/{name}"] = {
            "punctuation_boundaries": rnd(evaluate_boundaries(nb, test)), "tokens": rnd(evaluate_tokens(nb, test))}
    results["splitter"] = {"rule": rnd(sp.evaluate(sp.RuleSplitter(), test))}
    for name in sp.FEATURES:
        results["splitter"][f"logistic_regression/{name}"] = rnd(sp.evaluate(sp.LogRegSplitter(name).fit(train), test))
    print(json.dumps(results, indent=2, ensure_ascii=False))


def cmd_parse(args):
    from .conllu import read_conllu
    from .parsing.agreement import parse_with_agreement
    from .parsing.cky import CKYParser
    from .parsing.lexicon import load_lexicon

    parser = CKYParser(lexicon=load_lexicon(args.lexicon) if args.lexicon else None)
    for s in read_conllu(args.file):
        words = s.words
        trees, valid, rejected = parse_with_agreement(parser, words)
        sentence = " ".join(w.form for w in words)
        if not trees:
            status = "no parse"
        elif valid:
            status = f"accepted ({len(valid)} of {len(trees)} parses satisfy agreement)"
        else:
            status = "rejected: " + "; ".join(sorted({v for r in rejected for v in r}))
        print(f"{sentence}\n  {status}")
        for t in valid[: args.show]:
            print("  " + t.to_string(words))


def cmd_lexicon(args):
    from .parsing.lexicon import build_lexicon, save_lexicon

    n = save_lexicon(build_lexicon(args.conllu), args.out)
    print(f"{n} lemma-POS pairs written to {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="trnlp", description="Turkish text processing and CKY parsing")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, func, helptext in (
        ("tokenize", cmd_tokenize, "rule-based tokenization, one token per line"),
        ("split", cmd_split, "rule-based sentence splitting, one sentence per line"),
        ("normalize", cmd_normalize, "lower-case, expand abbreviations, drop punctuation"),
        ("stopwords", cmd_stopwords, "remove stopwords (static list or --dynamic)"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("file", nargs="?", help="text file (default: standard input)")
        s.set_defaults(func=func)
        if name == "stopwords":
            s.add_argument("--dynamic", action="store_true", help="use the most frequent words of the input")
            s.add_argument("--top-fraction", type=float, default=0.01)

    s = sub.add_parser("evaluate", help="train and evaluate tokenizers and sentence splitters on a UD treebank")
    s.add_argument("--train", required=True, help="training .conllu (e.g. tr_boun-ud-train.conllu)")
    s.add_argument("--test", required=True)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("parse", help="CKY-parse analysed sentences (.conllu) and check agreement")
    s.add_argument("file")
    s.add_argument("--lexicon", help="lemma<TAB>UPOS file for words without a tag")
    s.add_argument("--show", type=int, default=1, help="number of accepted trees to print")
    s.set_defaults(func=cmd_parse)

    s = sub.add_parser("lexicon", help="build a lemma-POS lexicon from UD treebanks")
    s.add_argument("conllu", nargs="+")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_lexicon)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
