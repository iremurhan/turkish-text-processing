# Turkish Text Processing and CKY Parsing

A small toolkit for Turkish: tokenization, sentence splitting, normalization and stopword elimination, each with a rule-based and (where it applies) a machine-learning version, plus a context-free grammar and CKY parser for simple SOV sentences that rejects sentences violating agreement constraints. Pure Python; NumPy and scikit-learn are used only by the logistic-regression sentence splitter.

Group project for CMPE 561 (Natural Language Processing) at Boğaziçi University, Fall 2023, refactored in 2026.

## Components

| Task | Rule-based | Machine learning |
|---|---|---|
| Tokenization | regular expressions for URLs, e-mails, hashtags, dates, times, currency, phone numbers, fractions, numbers, multi-word expressions, TDK abbreviations and ordinals | Naive Bayes boundary classifier, implemented from scratch |
| Sentence splitting | abbreviation, ordinal and closing-quote rules | logistic regression on hand-designed token features |
| Normalization | Turkish-aware lower-casing (I → ı, İ → i), abbreviation expansion, punctuation removal; special entities are kept intact | |
| Stopword elimination | static Turkish stopword list | dynamic list from corpus frequencies |
| Parsing | CNF grammar, CKY chart parser returning all parses, six agreement constraints | |

**Naive Bayes tokenizer.** Every whitespace or punctuation character is a candidate boundary. The classifier decides whether it separates tokens ("ressam ." vs "3.12.1999", "Ankara'da") from categorical features of its context: the character itself, one or two characters on each side, their capitalisation and character class, and whether a space follows. Training labels come from the treebank's tokens and `SpaceAfter` annotations. Add-one smoothing; four feature sets are compared.

**Logistic-regression sentence splitter.** Candidates are tokens ending in `.`, `!`, `?` or `…`. Features: token length, next-token capitalisation, numeric status (the 2023 "small" set); previous and next token lengths ("medium"); and, in the "full" set, abbreviation membership, a preceding number (ordinals), a following closing quote or bracket, a following lower-case word or digit, and end of input.

**Parser.** The input is morphologically analysed text in CoNLL-U form (lemma, part of speech, Universal Dependencies features), as if a morphological analyser and disambiguator had run first; copulas (*-dir*, *-ydı*), the question particle (*mı*) and the relativiser (*-ki*) are separate words. The grammar ([`trnlp/parsing/grammar.py`](trnlp/parsing/grammar.py)) covers declarative, interrogative and imperative SOV sentences without subordinate clauses. Every tree the CKY parser finds is checked against the agreement constraints in [`trnlp/parsing/agreement.py`](trnlp/parsing/agreement.py):

1. subject–verb person and number (a third-person plural subject may take a singular verb, as in *Destanlar … anlatır*);
2. determiner–noun number (*\*bir romanlar*);
3. time adverb and tense (*\*Dün … edeceğim*);
4. verb frames: no bare object for intransitive verbs, no locative argument for motion verbs (*\*Ben okul gittim*, *\*Ben okulda gittim*);
5. voice: no first- or second-person subject for a passive verb (*\*Ben kitap okundu*);
6. possessives: a pronoun modifying a noun must be genitive (*bizim köye*).

A sentence is accepted if at least one parse satisfies all constraints. Words given without a part of speech are looked up in a lemma–POS lexicon built from the Universal Dependencies Turkish treebanks (`python -m trnlp lexicon`).

## Data

- **Tokenizer and splitter:** [UD Turkish-BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN) (CC BY-SA 4.0), not included; download `tr_boun-ud-train.conllu` and `tr_boun-ud-test.conllu`.
- **Abbreviations:** 1,080 entries extracted from the Turkish Language Association's *Kısaltmalar Dizini* ([`trnlp/data/abbreviations.json`](trnlp/data/abbreviations.json)). Entries listing several senses are not used for expansion.
- **Stopwords:** a general-domain Turkish list of 219 words ([`trnlp/data/stopwords.txt`](trnlp/data/stopwords.txt)).
- **Parser sentences:** the example sentences of the assignment, analysed by hand, in [`data/sentences/`](data/sentences/).

## Usage

```bash
pip install -r requirements.txt

echo "Prof. Dr. Ayşe 3.12.1999'da geldi. Bu 5. sergi!" | python -m trnlp tokenize
echo "Prof. Dr. Ayşe 3.12.1999'da geldi. Bu 5. sergi!" | python -m trnlp split
echo "Prof. Dr. Ayşe 3.12.1999'da geldi. Bu 5. sergi!" | python -m trnlp normalize
python -m trnlp stopwords some_text.txt --dynamic

python -m trnlp evaluate --train tr_boun-ud-train.conllu --test tr_boun-ud-test.conllu
python -m trnlp parse data/sentences/well_formed.conllu --show 1
python -m trnlp parse data/sentences/ill_formed.conllu

pytest -q
```

```python
from trnlp.tokenizer import RuleTokenizer
from trnlp.splitter import RuleSplitter
from trnlp.normalizer import Normalizer

RuleTokenizer()("Kız Kulesi'ni 5. kez gördü!")   # ["Kız Kulesi'ni", '5.', 'kez', 'gördü', '!']
Normalizer()("Prof. Dr. Ali geldi.")              # 'profesör doktor ali geldi'
```

## Results

Trained on the BOUN training set (7,803 sentences) and evaluated on its test set (979 sentences).

**Tokenization.** Token scores count a predicted token as correct when its character span matches a treebank token. Boundary scores are over the 2,447 punctuation characters of the test set, for the "splits here" class.

| Model | Boundary P | Boundary R | Boundary F1 | Token F1 |
|---|---|---|---|---|
| Rule-based | – | – | – | 0.990 |
| Naive Bayes, character + neighbours | 0.990 | 0.995 | 0.993 | 0.993 |
| Naive Bayes, + capitalisation | 0.990 | 0.995 | 0.993 | 0.993 |
| Naive Bayes, ± 2 characters | 0.993 | 0.994 | 0.993 | 0.993 |
| Naive Bayes, all features | 0.993 | 0.994 | 0.994 | 0.993 |

Capitalisation of the neighbouring characters adds nothing on its own; two characters of context on each side lower the false splits from 21 to 16.

**Sentence splitting.** Scores for the "sentence ends here" class over the 933 candidate tokens of the test set.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Rule-based | 0.941 | 1.000 | 0.970 |
| Logistic regression, small (3 features) | 0.944 | 1.000 | 0.971 |
| Logistic regression, medium (5 features) | 0.943 | 0.999 | 0.970 |
| Logistic regression, full (12 features) | 0.959 | 0.998 | 0.978 |

Recall is near 1 for every model; the full feature set raises precision from 0.94 to 0.96.

**Parsing.** All ten example sentences of the assignment are accepted and all six ill-formed ones are rejected, each for the constraint it breaks:

| Sentence | Parses | Satisfy agreement |
|---|---|---|
| Dün arkadaşıma bir hediye aldım. | 2 | 2 |
| Tarihi romanları keyifle okuyorum. | 2 | 2 |
| Ben dün akşam yemeği için anneme yardım ettim. | 10 | 10 |
| Destanlar milli kültürümüzü ve tarihimizi anlatır. | 6 | 6 |
| Yaz meyvelerinden karpuz bence en güzel meyvedir. | 19 | 19 |
| Bu akşamki toplantıya katılacak mısınız? | 3 | 3 |
| Bu ağacın altında her gece mehtabı izlerdik. | 9 | 9 |
| Siz buraya en son ne zaman geldiniz? | 11 | 2 |
| Okul bizim köye epeyce uzaktaydı. | 10 | 10 |
| Yüksek sesle müzik dinleme. | 5 | 3 |
| \*Ben arkadaşıma hediye aldın. | 5 | 0 (person) |
| \*Tarihi bir romanlar okudum. | 2 | 0 (determiner number) |
| \*Dün babama yardım edeceğim. | 2 | 0 (tense) |
| \*Ben okul gittim. | 2 | 0 (verb frame) |
| \*Ben kitap okundu. | 2 | 0 (voice) |
| \*Ben okulda gittim. | 2 | 0 (verb frame) |

The parser is not probabilistic and returns every tree; the alternatives differ in where adverbials attach and in how the NP → NP NP rule groups nouns. `trnlp.parsing.evaluate` computes labelled bracket precision, recall and F1 against a reference tree; for the first sentence, the reference tree of the 2023 report is among the parses (F1 = 1.0).

## What changed from the 2023 version

- **Structure:** the Colab notebooks are a package (`trnlp/`) with a command-line interface and tests; Google Drive paths and Colab dependencies are gone.
- **Tokenizer:** the rule-based tokenizer also recognises TDK abbreviations, ordinals, times and apostrophe suffixes after special entities (*3.12.1999'da*). The Naive Bayes features are now named by position (left, right, ...) and smoothed, and training labels come from the treebank's `SpaceAfter` annotations.
- **Sentence splitter:** examples are the candidate tokens only, labelled by the treebank's sentence boundaries; features are standardised; the "full" feature set is new.
- **Normalizer:** Turkish-aware lower-casing and expansion only for abbreviations with a single sense.
- **Parser:** analysed CoNLL-U input with features, all parses with a memoised chart, and the agreement constraints, which were designed in 2023 and are now implemented.

## References

- D. Jurafsky and J. H. Martin. *Speech and Language Processing*, 3rd ed. draft. Chapters on tokenization, constituency grammars and CKY parsing.
- U. Türk, F. Atmaca, Ş. B. Özateş, G. Berk, S. T. Bedir, A. Köksal, B. Ö. Başaran, T. Güngör and A. Özgür. *Resources for Turkish Dependency Parsing: Introducing the BOUN Treebank and the BoAT Annotation Tool.* Language Resources and Evaluation, 2022.
- İ. Dönmez and E. Adalı. *Context-Free Grammar for Turkish.* Süleyman Demirel University Journal of Natural and Applied Sciences, 22(2), 2018.
- E. İ. Ünkar. *Parsing Turkish Sentences for Natural Language Watermarking.* B.S. thesis, Boğaziçi University, 2007.
- Türk Dil Kurumu. *Kısaltmalar Dizini.* 2019.
