"""
trust_engine/policy.py
=========================
Configurable fusion policy/thresholds for the Trust Decision Engine.

Kept separate from decision_engine.py so thresholds can be tuned (or
loaded from isce_config.yaml) without touching fusion logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from trust_engine.exceptions import InvalidPolicyConfigError
from trust_engine.models import SemanticRisk


@dataclass(frozen=True)
class TrustPolicy:
    """Thresholds governing how B1 + B2 + B3 combine into a FinalTrustDecision.

    semantic_high_confidence / semantic_medium_confidence / malicious_labels:
        FALLBACK ONLY. B3's own risk-banding is authoritative (see
        pipeline.b3_bridge.B3RiskPolicy) and is read via SemanticResult's
        "risk_level" field. These fields exist purely so
        classify_semantic_risk() can still degrade gracefully if it's
        ever handed a SemanticResult-shaped dict from an older/external
        source that predates the "risk_level" field. New code should not
        rely on this fallback path; it exists for robustness, not as the
        primary contract.

    cryptographic_reject_below / cryptographic_caution_below:
        B1 validation-score bands (non-fatal path), per the agreed
        fusion rules:
          - B1 fatal              -> REJECT, regardless of B3.
          - B1 pass + B3 high     -> REJECT (semantic override).
          - B1 pass + B3 medium   -> CAUTION (trust degraded).
          - B1 pass + B3 low/none -> ACCEPT.
    """

    semantic_high_confidence: float = 0.85
    semantic_medium_confidence: float = 0.60
    malicious_labels: frozenset = frozenset({"MALICIOUS", "MALICIOUS_SEMANTIC_MANIPULATION"})

    # Cryptographic/validation score bands (non-fatal B1 path). Mirrors the
    # legacy pipeline/fusion.py thresholds so a low, non-fatal B1 score
    # (e.g. stale timestamp, marginal cert rotation) still degrades trust
    # instead of silently defaulting to ACCEPT.
    cryptographic_reject_below: float = 0.40
    cryptographic_caution_below: float = 0.70

    def __post_init__(self) -> None:
        for name in (
            "semantic_high_confidence", "semantic_medium_confidence",
            "cryptographic_reject_below", "cryptographic_caution_below",
        ):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise InvalidPolicyConfigError(f"TrustPolicy.{name} must be in [0.0, 1.0], got {v}.")
        if self.semantic_medium_confidence >= self.semantic_high_confidence:
            raise InvalidPolicyConfigError(
                "TrustPolicy.semantic_medium_confidence must be < semantic_high_confidence "
                f"(got medium={self.semantic_medium_confidence}, high={self.semantic_high_confidence})."
            )
        if self.cryptographic_reject_below >= self.cryptographic_caution_below:
            raise InvalidPolicyConfigError(
                "TrustPolicy.cryptographic_reject_below must be < cryptographic_caution_below "
                f"(got reject_below={self.cryptographic_reject_below}, "
                f"caution_below={self.cryptographic_caution_below})."
            )

    @staticmethod
    def from_config(config: Optional[Dict[str, Any]]) -> "TrustPolicy":
        if not config:
            return TrustPolicy()
        te_cfg = config.get("trust_engine", {}) if isinstance(config, dict) else {}
        return TrustPolicy(
            semantic_high_confidence=te_cfg.get("semantic_high_confidence", 0.85),
            semantic_medium_confidence=te_cfg.get("semantic_medium_confidence", 0.60),
            cryptographic_reject_below=te_cfg.get("cryptographic_reject_below", 0.40),
            cryptographic_caution_below=te_cfg.get("cryptographic_caution_below", 0.70),
        )

    def classify_semantic_risk(self, b3_result: Dict[str, Any]) -> SemanticRisk:
        """Read B3's own risk_level (its public contract) as the primary
        source of truth. Falls back to computing from label+confidence
        ONLY if risk_level is absent, for backward compatibility with
        callers/tests predating the risk_level field.
        """
        if not b3_result.get("available", False):
            return SemanticRisk.UNAVAILABLE

        risk_level = b3_result.get("risk_level")
        if risk_level is not None:
            try:
                return SemanticRisk(risk_level)
            except ValueError:
                pass  # unrecognized string -> fall through to legacy path

        # Fallback path (no risk_level in the dict): compute it here.
        label = b3_result.get("label")
        confidence = b3_result.get("confidence")
        if label is None or confidence is None:
            return SemanticRisk.UNAVAILABLE

        if label not in self.malicious_labels:
            return SemanticRisk.NONE

        if confidence >= self.semantic_high_confidence:
            return SemanticRisk.HIGH
        if confidence >= self.semantic_medium_confidence:
            return SemanticRisk.MEDIUM
        return SemanticRisk.LOW
