"""Apply the frozen linear coefficients to a sparse feature matrix."""

import numpy as np


def scores(features, head) -> np.ndarray:
    return features.astype(np.float32, copy=False) @ head["w"] + head["b"]
