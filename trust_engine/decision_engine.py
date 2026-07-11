"""
trust_engine/decision_engine.py
=================================
TrustDecisionEngine: the only component that fuses B1 (ValidationAssessment),
B2 (ExplainabilityReport), and B3 (SemanticResult) into a FinalTrustDecision.

No B-layer imports another layer; this module is the sole composition
point, per the architecture's separation-of-concerns constraint.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from trust_engine.exceptions import MissingLayerInputError
from trust_engine.models import FinalTrustDecision, SemanticRisk, TrustLevel
from trust_engine.policy import TrustPolicy


class TrustDecisionEngine:
    def __init__(self, policy: Optional[TrustPolicy] = None) -> None:
        self.policy = policy or TrustPolicy()

    def decide(
        self,
        validation_assessment: Dict[str, Any],
        explainability_report: Dict[str, Any],
        semantic_result: Dict[str, Any],
    ) -> FinalTrustDecision:
        if validation_assessment is None:
            raise MissingLayerInputError("TrustDecisionEngine.decide: validation_assessment (B1) is required.")
        if explainability_report is None:
            raise MissingLayerInputError("TrustDecisionEngine.decide: explainability_report (B2) is required.")
        if semantic_result is None:
            raise MissingLayerInputError("TrustDecisionEngine.decide: semantic_result (B3) is required.")

        b1_fatal = bool(validation_assessment.get("fatal", False))
        b1_valid = bool(validation_assessment.get("valid", True))
        b1_score = float(explainability_report.get("validation_score", validation_assessment.get("score", 1.0)))
        b1_reasons = validation_assessment.get("reasons") or []

        semantic_risk = self.policy.classify_semantic_risk(semantic_result)

        # Rule 1: B1 fatal -> REJECT regardless of B3. B3 is explicitly not
        # consulted on this path, so it must not appear in contributors.
        if b1_fatal:
            return FinalTrustDecision(
                trust_score=0.0,
                trust_level=TrustLevel.REJECT,
                semantic_risk=SemanticRisk.UNAVAILABLE,
                cryptographic_risk="fatal",
                attack_detected=True,
                confidence=1.0,
                reasoning=(
                    f"B1 cryptographic/structural validation failed fatally "
                    f"({', '.join(b1_reasons) if b1_reasons else 'unspecified'}). "
                    f"B3 semantic result not consulted, per policy."
                ),
                contributors=["B1", "B2"],
                details={
                    "b1_score": b1_score,
                    "explanation": explainability_report.get("explanation_text"),
                },
            )

        contributors = ["B1", "B2"]
        if "CP" in explainability_report.get("provenance", {}).get("source_layers", []):
            contributors.append("CP")
        if semantic_result.get("available", False):
            contributors.append("B3")

        # Rule 2/3: B1 passed (possibly with non-fatal issues) -> weigh both
        # the cryptographic/validation score band AND B3's semantic risk,
        # taking the more conservative (lower-trust) of the two. This closes
        # a gap where a non-fatal but low-scoring/invalid B1 result (e.g.
        # stale timestamp, marginal cert rotation) would otherwise be
        # silently overridden to ACCEPT whenever B3 found nothing. Banding
        # is purely score-driven (not gated on the `valid` flag) so the
        # bands are monotonic and unambiguous.
        if b1_score < self.policy.cryptographic_reject_below:
            crypto_level, crypto_score = TrustLevel.REJECT, b1_score
            cryptographic_risk = "high"
        elif b1_score < self.policy.cryptographic_caution_below:
            crypto_level, crypto_score = TrustLevel.CAUTION, b1_score
            cryptographic_risk = "elevated"
        else:
            crypto_level, crypto_score = TrustLevel.ACCEPT, b1_score
            cryptographic_risk = "low"

        if semantic_risk == SemanticRisk.HIGH:
            semantic_level, semantic_score = TrustLevel.REJECT, 0.1
            attack_detected = True
        elif semantic_risk == SemanticRisk.MEDIUM:
            semantic_level, semantic_score = TrustLevel.CAUTION, 0.5
            attack_detected = False
        elif semantic_risk == SemanticRisk.LOW:
            semantic_level, semantic_score = TrustLevel.CAUTION, 0.7
            attack_detected = False
        else:  # NONE or UNAVAILABLE
            semantic_level, semantic_score = TrustLevel.ACCEPT, 1.0
            attack_detected = False

        _RANK = {TrustLevel.ACCEPT: 0, TrustLevel.CAUTION: 1, TrustLevel.REJECT: 2}
        if _RANK[crypto_level] >= _RANK[semantic_level]:
            trust_level, trust_score = crypto_level, crypto_score
        else:
            trust_level, trust_score = semantic_level, semantic_score

        crypto_note = (
            f"B1 non-fatal validation score {b1_score:.2f} -> {crypto_level.value} "
            f"(cryptographic_risk={cryptographic_risk})."
        )
        if semantic_risk in (SemanticRisk.NONE, SemanticRisk.UNAVAILABLE):
            b3_note = (
                "B3 unavailable, decision based on B1+B2 only."
                if semantic_risk == SemanticRisk.UNAVAILABLE
                else "B3 found no semantic risk."
            )
        else:
            b3_note = (
                f"B3 flagged {semantic_risk.value}-confidence semantic signal "
                f"(label={semantic_result.get('label')}, "
                f"confidence={semantic_result.get('confidence'):.2f}) -> {semantic_level.value}."
            )
        reasoning = f"{crypto_note} {b3_note} Final decision: {trust_level.value} (most conservative of the two)."

        return FinalTrustDecision(
            trust_score=trust_score,
            trust_level=trust_level,
            semantic_risk=semantic_risk,
            cryptographic_risk=cryptographic_risk,
            attack_detected=attack_detected,
            confidence=float(explainability_report.get("confidence_calibration", 1.0)),
            reasoning=reasoning,
            contributors=contributors,
            details={
                "b1_score": b1_score,
                "explanation": explainability_report.get("explanation_text"),
                "b3_label": semantic_result.get("label"),
                "b3_confidence": semantic_result.get("confidence"),
            },
        )
