from functools import lru_cache
from typing import Iterable, Optional, Any

import numpy as np


@lru_cache(maxsize=2)
def load_embedder(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """
    Load and cache the default sentence-transformer used across MBT-5 helpers.

    The original scripts relied on ``all-MiniLM-L6-v2`` for geometric
    measurements; this keeps the same model while loading the optional heavy
    dependency only when embeddings are actually requested.
    """

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(
    texts: Iterable[str],
    *,
    model_name: str = "all-MiniLM-L6-v2",
    embedder: Optional[Any] = None,
) -> np.ndarray:
    """
    Encode an iterable of texts into embeddings.

    Parameters mirror the notebook defaults and avoid altering any of the
    math or thresholds used downstream.
    """

    model = embedder or load_embedder(model_name)
    return np.asarray(model.encode(list(texts)))
