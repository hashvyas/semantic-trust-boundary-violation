"""
trust_engine/models.py
========================
Typed data models for the Trust Decision Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Optional


@unique
class TrustLevel(Enum):
    ACCEPT = "ACCEPT"
    CAUTION = "CAUTION"
    REJECT = "REJECT"


@unique
class SemanticRisk(Enum):
    """Normalized semantic risk level derived from B3's raw label/confidence.

    Keeping this here (rather than in b3_bridge) means the Trust Decision
    Engine only ever reasons about a fixed small taxonomy, decoupled from
    whatever label vocabulary B3's underlying model happens to emit.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class FinalTrustDecision:
    """The single unified output of the Trust Decision Engine."""

    trust_score: float
    trust_level: TrustLevel
    semantic_risk: SemanticRisk
    cryptographic_risk: str
    attack_detected: bool
    confidence: float
    reasoning: str
    contributors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trust_score": self.trust_score,
            "trust_level": self.trust_level.value,
            "semantic_risk": self.semantic_risk.value,
            "cryptographic_risk": self.cryptographic_risk,
            "attack_detected": self.attack_detected,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "contributors": self.contributors,
            "details": self.details,
        }
