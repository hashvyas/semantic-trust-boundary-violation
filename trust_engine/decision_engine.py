"""
trust_engine/decision_engine.py
=================================
TrustDecisionEngine: the only component that fuses B1 (ValidationAssessment),
B2 (ExplainabilityReport), and B3 (SemanticResult) into a FinalTrustDecision.

No B-layer imports another layer; this module is the sole composition
point, per the architecture's separation-of-concerns constraint.

FUSION RULE (roadmap A3/B1): the non-fatal path below combines the
crypto/structural evidence (B1, folded with MBD/CP via B2's
validation_score -- see pipeline/orchestrator.py) and B3's semantic
evidence via real Dempster-Shafer combination
(trust_engine/dempster_shafer.py), not a rank/threshold placeholder.
Each source is mapped to a mass function over {trustworthy, suspicious},
combined via Dempster's rule, and the combined mass is converted back to
a scalar trust score via the pignistic transform, banded against the
same thresholds already used for the crypto score alone (TrustPolicy),
so the decision rule is consistent whether or not B3 is available.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from trust_engine.dempster_shafer import MassFunction, combine
from trust_engine.exceptions import MissingLayerInputError
from trust_engine.models import FinalTrustDecision, SemanticRisk, TrustLevel
from trust_engine.policy import TrustPolicy

# Confidence clamp applied before building any mass function. Prevents a
# source from becoming "dogmatic" (m_theta == 0), which under Dempster's
# rule would give it absolute veto power over any conflicting evidence
# from the other source, however strong -- see
# trust_engine/dempster_shafer.py's module docstring for the full
# citation/explanation, and tests/test_dempster_shafer_fusion.py for the
# regression test proving this matters (without the clamp, a "clean" B1
# would silently discard a 99%-confident B3 MALICIOUS verdict).
MAX_SOURCE_CONFIDENCE = 0.98


class TrustDecisionEngine:

    def _semantic_mass(self, semantic_result: Dict[str, Any],
                        semantic_risk: SemanticRisk) -> MassFunction:
        """Build B3's mass function over {trustworthy, suspicious}.

        Two interfaces are supported. They agree exactly for
        p_malicious >= 0.5 and differ only below it.

        LEGACY (default; policy.use_continuous_semantic_belief == False)
            Consumes B3's argmax label + max-probability confidence. As a
            function of p = p(malicious), this yields:
                p <  0.5 -> (m_A, m_notA, m_theta) = (1-p, 0,   p)
                p >= 0.5 -> (m_A, m_notA, m_theta) = (0,   p, 1-p)
            i.e. below the decision boundary, semantic suspicion is routed
            into THETA (ignorance) rather than into DISBELIEF: the engine is
            told "I do not know" when B3 actually means "49% chance this is
            an attack". m_notA is discontinuous at p = 0.5 (jumps 0 -> 0.5).

        CONTINUOUS (opt-in; requires `p_malicious` on the SemanticResult)
            Consumes the calibrated probability directly:
                (m_A, m_notA, m_theta) = ((1-p)*c, p*c, 1-c)
            with c = MAX_SOURCE_CONFIDENCE, an explicit epistemic budget that
            keeps the source non-dogmatic (m_theta > 0 always -- see
            dempster_shafer.py on why a dogmatic source must never occur).
            This is the standard probability-to-mass conversion (a Bayesian
            mass function with a reserved ignorance budget). m_notA is
            continuous and monotone in p; sub-0.5 suspicion becomes graded
            disbelief instead of ignorance.

        NOTE the semantic trade-off, which is exactly what the A/B experiment
        measures: under CONTINUOUS, m_theta is CONSTANT (= 1-c), so B3's own
        uncertainty no longer inflates ignorance mass -- it is expressed
        entirely in p. Whether that is an improvement is an empirical
        question, not an aesthetic one. See INTERFACE_COMPARISON.md.
        """
        p_mal = semantic_result.get("p_malicious")
        if self.policy.use_continuous_semantic_belief and p_mal is not None:
            p = min(max(float(p_mal), 0.0), 1.0)
            c = MAX_SOURCE_CONFIDENCE
            return MassFunction(m_A=(1.0 - p) * c, m_not_A=p * c, m_theta=1.0 - c)

        raw_confidence = semantic_result.get("confidence")
        if raw_confidence is None:
            # No raw confidence on this SemanticResult (e.g. an older/
            # external caller supplying only risk_level) -- fall back to a
            # representative confidence per band so fusion still has
            # something principled to combine, rather than silently
            # treating it as vacuous.
            raw_confidence = {"high": 0.90, "medium": 0.65,
                              "low": 0.40, "none": 0.50}[semantic_risk.value]
        semantic_confidence = min(float(raw_confidence), MAX_SOURCE_CONFIDENCE)
        # NONE -> supports "trustworthy"; LOW/MEDIUM/HIGH -> supports
        # "suspicious", with the actual model confidence (not a fixed
        # per-band score) driving how much mass is committed.
        semantic_score_for_mass = 1.0 if semantic_risk == SemanticRisk.NONE else 0.0
        return MassFunction.from_score_confidence(score=semantic_score_for_mass,
                                                   confidence=semantic_confidence)

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

        # cryptographic_risk (reported field) still describes the crypto/
        # structural layer's own standing, independent of fusion -- this
        # banding is unchanged from before.
        if b1_score < self.policy.cryptographic_reject_below:
            cryptographic_risk = "high"
        elif b1_score < self.policy.cryptographic_caution_below:
            cryptographic_risk = "elevated"
        else:
            cryptographic_risk = "low"

        # === Dempster-Shafer fusion (roadmap A3/B1) ===
        # Map each source to a mass function over {trustworthy, suspicious},
        # clamping confidence below 1.0 so neither source can become
        # dogmatic (see MAX_SOURCE_CONFIDENCE's docstring above).
        crypto_confidence = min(float(explainability_report.get("confidence_calibration", 1.0)), MAX_SOURCE_CONFIDENCE)
        crypto_mass = MassFunction.from_score_confidence(score=b1_score, confidence=crypto_confidence)

        if semantic_risk == SemanticRisk.UNAVAILABLE:
            semantic_mass = MassFunction.vacuous()
        else:
            semantic_mass = self._semantic_mass(semantic_result, semantic_risk)

        conflict_mass = crypto_mass.conflict_with(semantic_mass)
        fused_mass = combine(crypto_mass, semantic_mass)
        trust_score = fused_mass.pignistic_trust_score()

        crypto_trust_score = crypto_mass.pignistic_trust_score()
        if crypto_trust_score < self.policy.cryptographic_reject_below:
            crypto_level_alone = TrustLevel.REJECT
        elif crypto_trust_score < self.policy.cryptographic_caution_below:
            crypto_level_alone = TrustLevel.CAUTION
        else:
            crypto_level_alone = TrustLevel.ACCEPT

        # Floor rule: if any upstream validation failed (excluding pure stale timestamps), the decision must be at least CAUTION
        if not explainability_report.get("validation_valid", True) or not b1_valid:
            reasons = validation_assessment.get("reasons") or []
            is_only_stale_ts = len(reasons) == 1 and any("stale" in r.lower() or "timestamp" in r.lower() for r in reasons)
            if not is_only_stale_ts:
                if crypto_level_alone == TrustLevel.ACCEPT:
                    crypto_level_alone = TrustLevel.CAUTION

        if trust_score < self.policy.cryptographic_reject_below:
            fused_level = TrustLevel.REJECT
        elif trust_score < self.policy.cryptographic_caution_below:
            fused_level = TrustLevel.CAUTION
        else:
            fused_level = TrustLevel.ACCEPT

        # Conservative-bias ceiling: B3 (or CP folded into crypto) may make
        # the decision MORE cautious than crypto alone, never less. Absence
        # of semantic risk (B3 "NONE") must not inflate trust past what
        # crypto/structural evidence already earned -- this preserves the
        # system's conservative-bias design property while still using
        # real DS fusion (not a hand-picked constant) to determine how
        # much worse things get when semantic evidence IS unfavorable.
        _RANK = {TrustLevel.ACCEPT: 0, TrustLevel.CAUTION: 1, TrustLevel.REJECT: 2}
        trust_level = crypto_level_alone if _RANK[crypto_level_alone] >= _RANK[fused_level] else fused_level

        # --- Explicit policy floors on top of the fused score ---
        # Yager's rule correctly reports strong disagreement as elevated
        # uncertainty (see dempster_shafer.py's module docstring), but a
        # symmetric evidence combination alone will not reliably push a
        # confident semantic-attack signal all the way to REJECT against
        # a confident "looks clean" crypto signal -- and for this threat
        # model it must. These floors are deliberate, asymmetric-cost
        # policy decisions layered on top of the fusion math (mirroring
        # the pre-existing B1-fatal short-circuit above, which is the
        # same kind of override), not a claimed property of Yager's rule
        # itself:
        #   - B3 HIGH-confidence semantic risk floors the decision at
        #     REJECT: a confirmed high-confidence attack signal must not
        #     be softened by conflicting-but-less-decisive crypto evidence.
        #   - B3 MEDIUM/LOW-confidence semantic risk floors the decision
        #     at (at least) CAUTION: a real, if less certain, semantic
        #     concern must never be silently smoothed away to ACCEPT.
        if semantic_risk == SemanticRisk.HIGH:
            trust_level = TrustLevel.REJECT
        elif semantic_risk in (SemanticRisk.MEDIUM, SemanticRisk.LOW):
            if _RANK[trust_level] < _RANK[TrustLevel.CAUTION]:
                trust_level = TrustLevel.CAUTION
        attack_detected = trust_level == TrustLevel.REJECT

        crypto_note = (
            f"B1(+MBD/CP) crypto/structural mass: m_A={crypto_mass.m_A:.2f} "
            f"m_not_A={crypto_mass.m_not_A:.2f} m_theta={crypto_mass.m_theta:.2f} "
            f"(validation_score={b1_score:.2f}, cryptographic_risk={cryptographic_risk})."
        )
        if semantic_risk in (SemanticRisk.NONE, SemanticRisk.UNAVAILABLE):
            b3_note = (
                "B3 unavailable (vacuous mass, does not affect fusion)."
                if semantic_risk == SemanticRisk.UNAVAILABLE
                else f"B3 found no semantic risk (mass: m_A={semantic_mass.m_A:.2f} m_theta={semantic_mass.m_theta:.2f})."
            )
        else:
            b3_note = (
                f"B3 flagged {semantic_risk.value}-confidence semantic signal "
                f"(label={semantic_result.get('label')}, confidence={semantic_result.get('confidence')}) "
                f"-> mass m_not_A={semantic_mass.m_not_A:.2f} m_theta={semantic_mass.m_theta:.2f}."
            )
        reasoning = (
            f"{crypto_note} {b3_note} Dempster combination: conflict K={conflict_mass:.3f}, "
            f"fused mass m_A={fused_mass.m_A:.2f} m_not_A={fused_mass.m_not_A:.2f} m_theta={fused_mass.m_theta:.2f}, "
            f"pignistic trust_score={trust_score:.3f}. Final decision: {trust_level.value}."
        )

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
                "ds_conflict_K": conflict_mass,
                "ds_fused_mass": {"m_A": fused_mass.m_A, "m_not_A": fused_mass.m_not_A, "m_theta": fused_mass.m_theta},
                "ds_crypto_mass": {"m_A": crypto_mass.m_A, "m_not_A": crypto_mass.m_not_A, "m_theta": crypto_mass.m_theta},
                "ds_semantic_mass": {"m_A": semantic_mass.m_A, "m_not_A": semantic_mass.m_not_A, "m_theta": semantic_mass.m_theta},
            },
        )
