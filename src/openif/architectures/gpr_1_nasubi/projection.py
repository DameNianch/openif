"""Generate the input rows and multipliers for the serialized feature blocks."""

from itertools import islice, pairwise

from sudachipy.errors import SudachiError

from .morphology import morphology, normalize, segmented_words
from .rules import CONCEPTS, CONTENT, LEXICON, STOP, category, evidence

# These keys are part of the fitted vectorizer format. Column order comes from the model.
WEIGHTS = {
    **dict.fromkeys(LEXICON, 0.42),
    **dict.fromkeys(("B018", "B020", "B036", "B076", "B078", "B080"), 0.52),
    "B066": 0.58,
    "B067": 0.58,
    "B104": 0.55,
    "readop": 0.42,
    "char@rare": 1.0,
    "lemma@rare": 1.0,
    "kana3": 0.42,
    "compoundop": 0.42,
}


def grams(values):
    return values + [a + "␟" + b for a, b in pairwise(values)]


def feature_rows(text: str) -> dict[str, list[str] | str]:
    try:
        tokens = segmented_words(text)
        fine, compounds, alternate = morphology(text)
    except SudachiError as error:
        raise ValueError(
            f"Sudachi could not tokenize this utterance: {error}"
        ) from error
    lemmas = [t.lemma for t in tokens]
    surfaces = [t.surface for t in tokens]
    content = [t.lemma for t in tokens if t.pos in CONTENT and t.lemma not in STOP]
    nouns = [t for t in tokens if t.pos in ("名詞", "代名詞")]
    predicates = [
        t for t in tokens if t.pos in ("動詞", "形容詞") and t.lemma not in STOP
    ]
    categories = [category(t) for t in tokens]
    operators = [(t, c) for t, c in zip(tokens, categories) if c in CONCEPTS]
    canonical = [t.normalized for t in fine]

    rows = {key: evidence(spec, lemmas, surfaces) for key, spec in LEXICON.items()}
    rows.update(
        {
            "B018": [
                a + "~" + b for gap in (2, 3) for a, b in zip(content, content[gap:])
            ],
            "B020": [
                word[i : i + n]
                for word in content
                for n in (1, 2, 3)
                for i in range(len(word) - n + 1)
            ],
            "B036": list(
                islice(
                    (
                        a.lemma + "/" + b.lemma + "/" + v.lemma
                        for a, b in pairwise(tokens)
                        if b.pos == "助詞"
                        for v in predicates
                        if v.start > b.start
                    ),
                    30,
                )
            ),
            "B066": grams(
                [
                    t.reading if t.pos in ("名詞", "形容詞") else t.normalized
                    for t in fine
                ]
            ),
            "B067": [
                c + "/" + min(nouns, key=lambda n: abs(n.start - t.start)).lemma
                for t, c in operators
                if nouns
            ],
            "B076": grams(categories),
            "B078": [category(v) + "/" + n.lemma for v in predicates for n in nouns],
            "B080": [
                t.lemma + "/" + v.lemma
                for t, c in zip(tokens, categories)
                if t.pos == "助動詞" or c in ("want", "negative")
                for v in predicates
            ],
            "B104": (
                ["u/" + w for w in alternate]
                + ["s/" + w for w in canonical]
                + ["agree/" + w for w in set(alternate) & set(canonical)]
            ),
            "readop": [
                c + "/" + m.reading
                for _, c in operators
                for m in fine
                if m.pos == "名詞"
            ],
            "char@rare": normalize(text),
            "lemma@rare": grams(lemmas),
            "kana3": [
                m.reading[i : i + 3]
                for m in fine
                if m.pos in CONTENT
                for i in range(len(m.reading) - 2)
            ],
            "compoundop": [
                c + "/" + w for _, c in operators for w in compounds if len(w) > 1
            ],
        }
    )
    return rows
