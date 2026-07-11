"""
b2_explain/evidence.py
========================
Evidence primitives used by ExplainabilityReport. Split out from models.py
so evidence-construction concerns (direction taxonomy, weighting) are not
mixed with the top-level report/serialization model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional


@unique
class EvidenceDirection(Enum):
    """Whether a piece of evidence supports or contradicts B1's verdict."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class EvidenceItem:
    """A single interpretable piece of evidence drawn from B1's output.

    Parameters
    ----------
    factor:
        Name of the B1 check/field this evidence is derived from
        (e.g. ``"timestamp_freshness"``, ``"kinematic_plausibility"``).
    description:
        Human-readable explanation of what this factor showed.
    weight:
        Relative importance of this factor in [0.0, 1.0].
    direction:
        Whether this factor supports, contradicts, or is neutral with
        respect to B1's ``valid`` verdict.
    source_field:
        The specific key in B1's ``checks``/``details``/``reasons`` this
        evidence was extracted from, for provenance/auditing.
    """

    factor: str
    description: str
    weight: float
    direction: EvidenceDirection
    source_field: Optional[str] = None
