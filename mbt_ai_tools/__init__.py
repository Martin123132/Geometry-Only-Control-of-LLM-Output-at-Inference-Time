"""
Public entrypoint for ManifoldGuard reference-bounded output regulation tools.

The package keeps the historical ``mbt_ai_tools`` import path for compatibility
while exposing the ManifoldGuard product name through packaging and CLI entry
points.
"""

from importlib.metadata import PackageNotFoundError, version

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

__version__ = "0.1.6"
for _distribution_name in ("manifold-guard", "mbt-ai-tools"):
    try:
        __version__ = version(_distribution_name)
        break
    except PackageNotFoundError:
        pass

__all__ = [
    "__version__",
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
