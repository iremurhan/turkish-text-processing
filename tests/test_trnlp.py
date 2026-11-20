from pathlib import Path

import pytest

from trnlp.conllu import read_conllu
from trnlp.normalizer import Normalizer, turkish_lower
from trnlp.parsing.agreement import parse_with_agreement
from trnlp.parsing.cky import CKYParser
from trnlp.parsing.evaluate import bracket_scores, read_bracketed
from trnlp.splitter import RuleSplitter
from trnlp.stopwords import dynamic_stopwords, remove_stopwords
from trnlp.tokenizer import NaiveBayesTokenizer, RuleTokenizer, boundary_examples

DATA = Path(__file__).resolve().parent.parent / "data" / "sentences"

CONLLU = """# text = Ali 3.12.1999'da geldi.
1\tAli\tAli\tPROPN\t_\t_\t_\t_\t_\t_
2\t3.12.1999'da\t3.12.1999\tNUM\t_\t_\t_\t_\t_\t_
3-4\tgeldi\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No
3\tgel\tgel\tVERB\t_\t_\t_\t_\t_\t_
4\tdi\ti\tAUX\t_\t_\t_\t_\t_\t_
5\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""


@pytest.fixture
def treebank(tmp_path):
    path = tmp_path / "mini.conllu"
    path.write_text(CONLLU * 20, encoding="utf-8")
    return read_conllu(path)


def test_conllu_keeps_surface_tokens_and_spacing(treebank):
    s = treebank[0]
    assert s.tokens == ["Ali", "3.12.1999'da", "geldi", "."]
    assert s.text == "Ali 3.12.1999'da geldi."
    assert [w.form for w in s.words] == ["Ali", "3.12.1999'da", "gel", "di", "."]


def test_rule_tokenizer_special_entities():
    tok = RuleTokenizer()
    text = "Prof. Dr. Ayşe www.ornek.com.tr adresinde 3.12.1999'da ₺155.130,25 ödedi; Kız Kulesi'ni gördü, 5. kez!"
    tokens = tok(text)
    for expected in ("Prof.", "Dr.", "www.ornek.com.tr", "3.12.1999'da", "₺155.130,25", "Kız Kulesi'ni", "5.", "!"):
        assert expected in tokens


def test_boundary_labels(treebank):
    labels = {(text[i], i): y for text, i, y in boundary_examples(treebank[0])}
    text = treebank[0].text
    assert labels[(".", text.index("."))] is False   # inside the date
    assert labels[(".", len(text) - 1)] is True       # sentence-final full stop
    assert labels[("'", text.index("'"))] is False


def test_naive_bayes_tokenizer_learns_boundaries(treebank):
    nb = NaiveBayesTokenizer("full").fit(treebank)
    assert nb("Ali 3.12.1999'da geldi.") == ["Ali", "3.12.1999'da", "geldi", "."]


def test_rule_splitter_abbreviations_and_ordinals():
    sentences = RuleSplitter()("Dr. Ayşe geldi. Bu 5. sergi! Çok güzel… Değil mi?")
    assert sentences == ["Dr. Ayşe geldi .", "Bu 5. sergi !", "Çok güzel …", "Değil mi ?"]


def test_normalizer():
    assert turkish_lower("IŞIK İzmir") == "ışık izmir"
    out = Normalizer()("Prof. Dr. Ali www.site.com adresinde!")
    assert out == "profesör doktor ali www.site.com adresinde"


def test_stopwords():
    tokens = ["bu", "kitap", "çok", "güzel", "ve", "bu", "film", "de"]
    assert remove_stopwords(tokens) == ["kitap", "güzel", "film"]
    assert dynamic_stopwords(tokens, top_fraction=0.1) == {"bu"}


@pytest.mark.parametrize("sentence", read_conllu(DATA / "well_formed.conllu"), ids=lambda s: s.words[0].form)
def test_parser_accepts_well_formed(sentence):
    _, valid, _ = parse_with_agreement(CKYParser(), sentence.words)
    assert valid


@pytest.mark.parametrize("sentence", read_conllu(DATA / "ill_formed.conllu"), ids=lambda s: " ".join(w.form for w in s.words))
def test_parser_rejects_ill_formed(sentence):
    trees, valid, _ = parse_with_agreement(CKYParser(), sentence.words)
    assert trees and not valid


def test_gold_tree_is_among_parses():
    s = read_conllu(DATA / "well_formed.conllu")[0]
    gold = read_bracketed("[S [S1 [ADV Dün] [VP [NP arkadaşıma] [VP [NP [DET bir] [NP hediye]] [VP aldım]]]] [PUNCT .]]")
    best = max(bracket_scores(t.constituents(), gold)["f1"] for t in CKYParser().trees(s.words))
    assert best == 1.0


def test_lexicon_fallback_for_untagged_words():
    from trnlp.conllu import Word

    words = [Word("Ali", "ali", "_", {}), Word("geldi", "gel", "_", {}), Word(".", ".", "PUNCT", {})]
    parser = CKYParser(lexicon={"ali": {"PROPN"}, "gel": {"VERB"}})
    assert parser.trees(words)
