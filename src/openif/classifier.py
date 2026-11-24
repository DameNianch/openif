"""Public interface for Japanese intent classification."""

from __future__ import annotations

from pathlib import Path
from typing import overload

import numpy as np

from .architectures.gpr_1_nasubi.extractor import Extractor
from .architectures.gpr_1_nasubi.head import scores
from .hub import NASUBI, load_model


class Classifier:
    """Classify raw Japanese utterances using the official Nasubi model."""

    model_id = NASUBI.model_id
    architecture = NASUBI.architecture
    extractor_version = Extractor.version

    def __init__(self, model: dict):
        self._extractor = Extractor(model["keys"], model["vec"])
        self.labels: tuple[str, ...] = tuple(model["labels"])
        self._head = model["head"]
        if (
            set(self._head) != {"w", "b"}
            or self._extractor.version != NASUBI.extractor_version
            or self._extractor.dimensions != NASUBI.dimensions
            or len(self.labels) != NASUBI.label_count
            or len(set(self.labels)) != NASUBI.label_count
            or self._head["w"].shape != (NASUBI.dimensions, NASUBI.label_count)
            or self._head["b"].shape != (NASUBI.label_count,)
        ):
            raise ValueError(
                "Model dimensions, labels or head are incompatible with Nasubi."
            )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
    ) -> Classifier:
        """Fetch or reuse the pinned official artifact and verify it before decoding."""
        return cls(load_model(model_id, revision, cache_dir, local_files_only))

    def decision_function(self, texts: str | list[str]) -> np.ndarray:
        """Return float32 margins in labels order, shaped (60,) or (n, 60)."""
        single = isinstance(texts, str)
        items = [texts] if single else texts
        if not isinstance(items, list) or not all(isinstance(t, str) for t in items):
            raise TypeError("Pass a string or a list containing only strings.")
        result = np.empty((len(items), len(self.labels)), dtype=np.float32)
        for start in range(0, len(items), 128):
            batch = items[start : start + 128]
            result[start : start + len(batch)] = scores(
                self._extractor.transform(batch), self._head
            )
        return result[0] if single else result

    @overload
    def predict(self, texts: str) -> str: ...

    @overload
    def predict(self, texts: list[str]) -> list[str]: ...

    def predict(self, texts: str | list[str]) -> str | list[str]:
        """Return one intent per utterance, preserving order and duplicates."""
        margins = self.decision_function(texts)
        if isinstance(texts, str):
            return self.labels[int(margins.argmax())]
        return [self.labels[i] for i in margins.argmax(axis=1)]
