"""
Public entrypoint for MBT-5 geometry-only tools.

The functions exposed here include the original stability helpers plus the
reference-bound MBT-5 v11 candidate-regulation API.
"""

from .mbt import (  # noqa: F401
    ManifoldRegulator,
    ZeroBoxConsensus,
    classify_entropy,
    confidence_score,
    hallucination_risk,
    token_shock_map,
    CandidateEvaluation,
    ReferenceManifold,
    RegulationResult,
    evaluate_candidate,
    extract_relations,
    regulate_candidates,
)

__all__ = [
    "confidence_score",
    "hallucination_risk",
    "token_shock_map",
    "ManifoldRegulator",
    "ZeroBoxConsensus",
    "classify_entropy",
    "CandidateEvaluation",
    "ReferenceManifold",
    "RegulationResult",
    "evaluate_candidate",
    "extract_relations",
    "regulate_candidates",
]
