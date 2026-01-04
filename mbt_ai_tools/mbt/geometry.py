from typing import Iterable

import numpy as np


def geometric_median(vectors: Iterable[np.ndarray], iterations: int = 30) -> np.ndarray:
    """
    Robust geometric median used across the original MBT-5 scripts.

    This mirrors the Weiszfeld-style update loop from the Colab demos without
    modifying thresholds or behavior.
    """

    vectors = np.asarray(list(vectors))
    if len(vectors) == 1:
        return vectors[0]

    y = np.mean(vectors, axis=0)
    eps = 1e-5

    for _ in range(iterations):
        d = np.maximum(np.linalg.norm(vectors - y, axis=1), eps)
        y = np.sum(vectors * (1 / d)[:, None], axis=0) / np.sum(1 / d)

    return y


def shock(embedding: np.ndarray, center: np.ndarray) -> float:
    """
    Measure squared radial escape from a manifold center.

    This preserves the ``np.linalg.norm(... ) ** 2`` formulation in the
    notebooks.
    """

    return float(np.linalg.norm(embedding - center) ** 2)


def mean_squared_distance(vectors: Iterable[np.ndarray], center: np.ndarray) -> float:
    """
    Compute average squared distance from a supplied center.

    The default center in the notebooks was either a median or manifold anchor;
    callers should provide the same to avoid behavior changes.
    """

    vectors = np.asarray(list(vectors))
    return float(np.mean([np.linalg.norm(v - center) ** 2 for v in vectors]))
