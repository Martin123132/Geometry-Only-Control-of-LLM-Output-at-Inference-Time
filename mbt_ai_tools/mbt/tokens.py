from typing import List, Tuple

import numpy as np

from .embeddings import embed_texts
from .geometry import shock


def token_shock_map(text: str) -> List[Tuple[str, float]]:
    """
    Compute leave-one-out token shock scores.

    This mirrors the notebook logic of removing words individually, embedding
    the modified text, and measuring squared deviation from the original
    embedding. No thresholds or new behaviors are introduced.
    """

    tokens = text.split()
    if not tokens:
        return []

    baseline_embedding = embed_texts([text])[0]
    scores: List[Tuple[str, float]] = []

    for i, token in enumerate(tokens):
        modified = " ".join(tokens[:i] + tokens[i + 1 :])
        modified_embedding = embed_texts([modified])[0]
        scores.append((token, shock(modified_embedding, baseline_embedding)))

    return scores
