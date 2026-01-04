"""
Public entrypoint for MBT-5 geometry-only tools.

The functions exposed here mirror the notebook behaviors without adding new
abstractions or altering thresholds.
"""

from .mbt import (  # noqa: F401
    ManifoldRegulator,
    ZeroBoxConsensus,
    classify_entropy,
    confidence_score,
    hallucination_risk,
    token_shock_map,
)

__all__ = [
    "confidence_score",
    "hallucination_risk",
    "token_shock_map",
    "ManifoldRegulator",
    "ZeroBoxConsensus",
    "classify_entropy",
]
