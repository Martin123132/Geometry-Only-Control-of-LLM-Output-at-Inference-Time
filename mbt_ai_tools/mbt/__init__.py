"""
MBT-5 geometry-only inference-time regulator public API.

This package exposes light-weight helpers for semantic stability, geometric
shock, and reference-bound MBT-5 v11 candidate regulation.
"""

from .embeddings import load_embedder, embed_texts
from .geometry import geometric_median, shock, mean_squared_distance
from .stability import (
    internal_entropy,
    classify_entropy,
    confidence_score,
    hallucination_risk,
)
from .tokens import token_shock_map
from .consensus import ManifoldRegulator, ZeroBoxConsensus
from .regulator import (
    CandidateEvaluation,
    ReferenceManifold,
    RegulationResult,
    evaluate_candidate,
    extract_relations,
    regulate_candidates,
)

__all__ = [
    "load_embedder",
    "embed_texts",
    "geometric_median",
    "shock",
    "mean_squared_distance",
    "internal_entropy",
    "classify_entropy",
    "confidence_score",
    "hallucination_risk",
    "token_shock_map",
    "ManifoldRegulator",
    "ZeroBoxConsensus",
    "CandidateEvaluation",
    "ReferenceManifold",
    "RegulationResult",
    "evaluate_candidate",
    "extract_relations",
    "regulate_candidates",
]
