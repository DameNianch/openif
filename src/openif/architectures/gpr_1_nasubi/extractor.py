"""Transform Japanese text with the feature vocabulary and IDF supplied by the model."""

from scipy import sparse

from .projection import WEIGHTS, feature_rows


class Extractor:
    version = "1"

    def __init__(self, keys, vectorizers):
        if (
            len(keys) != len(WEIGHTS)
            or set(keys) != WEIGHTS.keys()
            or set(vectorizers) != set(keys)
        ):
            raise ValueError("Incompatible Nasubi feature blocks.")
        self.blocks = tuple((k, vectorizers[k], WEIGHTS[k]) for k in keys)
        self.dimensions = sum(len(v.vocabulary_) for _, v, _ in self.blocks)

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        rows = [feature_rows(text) for text in texts]
        columns = [
            v.transform([row[key] for row in rows]) * weight
            for key, v, weight in self.blocks
        ]
        return sparse.hstack(columns, format="csr")
