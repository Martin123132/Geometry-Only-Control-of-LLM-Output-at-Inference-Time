from typing import Iterable, List, Tuple

import numpy as np

from .embeddings import embed_texts
from .geometry import mean_squared_distance

# Thresholds taken directly from the Zero-Box stability pilot.
HIGH_CONFIDENCE_MAX = 0.12
LOW_CONFIDENCE_MAX = 0.30


def internal_entropy(responses: Iterable[str]) -> float:
    """
    Measure self-disagreement across multiple responses.

    Mirrors ``MBT5StabilityCore.internal_entropy`` from the original notebook by
    embedding all responses, taking the coordinate-wise median, and returning
    the mean squared distance to that center.
    """

    responses = list(responses)
    if not responses:
        return 0.0

    embeddings = embed_texts(responses)
    center = np.median(embeddings, axis=0)
    return mean_squared_distance(embeddings, center)


def classify_entropy(entropy: float) -> Tuple[str, str]:
    """
    Classify stability using the same labels and thresholds from the notebook.

    Returns (label, color_hex) to keep CLI and UI behavior aligned.
    """

    if entropy < HIGH_CONFIDENCE_MAX:
        return "✅ HIGH CONFIDENCE", "#00ff88"
    if entropy < LOW_CONFIDENCE_MAX:
        return "⚠️ LOW CONFIDENCE OUTPUT", "#ffaa00"
    return "🚨 UNSTABLE / POSSIBLE HALLUCINATION", "#ff3333"


def _extract_responses(prompt: str) -> List[str]:
    """
    Extract candidate responses from a string.

    The notebooks supply multiple answer variants from the model. To preserve
    that behavior without forcing network calls, this helper treats blank-line
    separated blocks as distinct responses. A single block mirrors the
    single-sample path in the original code, yielding zero entropy.
    """

    parts = [p.strip() for p in prompt.split("\n\n") if p.strip()]
    return parts or [prompt]


def confidence_score(prompt: str) -> float:
    """
    Public helper: compute the MBT-5 internal entropy score for a prompt.

    If the prompt contains multiple blank-line separated responses, each block
    is treated as a distinct sample exactly as the Zero-Box pilot expects.
    """

    responses = _extract_responses(prompt)
    return internal_entropy(responses)


def hallucination_risk(prompt: str) -> dict:
    """
    Return a structured hallucination risk summary.

    Mirrors the Zero-Box pilot classification without adding new behaviors.
    """

    score = confidence_score(prompt)
    label, color = classify_entropy(score)
    return {"score": score, "label": label, "color": color}
