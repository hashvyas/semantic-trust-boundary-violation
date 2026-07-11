"""
b2_explain/explainability.py
=============================
ExplainabilityEngine: the B2 layer.

Contract
--------
Input:  a ValidationAssessment dict as produced by B1 (SCSV), i.e. the
        ``{"valid", "fatal", "score", "confidence", "reasons", "checks",
        "details"}`` shape already normalized by pipeline/orchestrator.py.
Output: an ExplainabilityReport.

Hard constraints (do not violate):
* B2 operates EXCLUSIVELY on ValidationAssessment and its associated
  validation metadata (checks/details/reasons). It never receives or
  interprets the raw V2X payload / message text.
* B2 NEVER performs semantic reasoning. That is B3's exclusive domain.
* B2 NEVER changes B1's verdict (valid/fatal/score). It only explains it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from b2_explain.config import B2Config
from b2_explain.evidence import EvidenceDirection, EvidenceItem
from b2_explain.models import ExplainabilityReport
from contracts.trust_evidence import TrustEvidence


class ExplainabilityEngine:
    """Produces an ExplainabilityReport from a ValidationAssessment only."""

    def __init__(self, config: Optional[B2Config] = None) -> None:
        self.config = config or B2Config()

    def explain_evidence(self, evidence: List[TrustEvidence]) -> ExplainabilityReport:
        """Explains a list of TrustEvidence (B1 + MBD + CP, per the frozen
        architecture) as a single unified report. UNCHANGED contract
        constraints apply: never modifies any upstream score/verdict,
        never inspects raw payload, never performs semantic reasoning.

        This is additive alongside explain() (which remains for B1-only
        callers) -- not a replacement. See b2_explain module docstring
        / prior architecture design doc §3.2 for why this is additive,
        not a breaking change.
        """
        if not evidence:
            raise ValueError(
                "ExplainabilityEngine.explain_evidence: at least one "
                "TrustEvidence item is required."
            )

        overall_passed = all(e.passed for e in evidence)
        overall_score = sum(e.score for e in evidence) / len(evidence)
        overall_confidence = sum(e.confidence for e in evidence) / len(evidence)

        items: List[EvidenceItem] = []
        for e in evidence:
            weight = 1.0 / len(evidence)
            if e.findings:
                for finding in e.findings:
                    items.append(
                        EvidenceItem(
                            factor=f"{e.source_layer}:{finding[:60]}",
                            description=finding,
                            weight=weight / max(len(e.findings), 1),
                            direction=(
                                EvidenceDirection.SUPPORTS
                                if e.passed
                                else EvidenceDirection.CONTRADICTS
                            ),
                            source_field=e.source_layer,
                        )
                    )
            else:
                items.append(
                    EvidenceItem(
                        factor=f"{e.source_layer}_no_findings",
                        description=f"{e.source_layer} reported no specific findings "
                                    f"(score={e.score:.2f}, passed={e.passed}).",
                        weight=weight,
                        direction=(
                            EvidenceDirection.SUPPORTS
                            if e.passed
                            else EvidenceDirection.NEUTRAL
                        ),
                        source_field=e.source_layer,
                    )
                )

        narrative_parts = []
        for e in evidence:
            verdict = "passed" if e.passed else "flagged an issue"
            narrative_parts.append(
                f"{e.source_layer} {verdict} (score={e.score:.2f}, confidence={e.confidence:.2f})"
            )
        explanation_text = (
            f"Combined trust evidence from {len(evidence)} upstream layer(s): "
            + "; ".join(narrative_parts) + ". "
            + ("All layers agree this message is trustworthy so far."
               if overall_passed else
               "At least one upstream layer flagged a concern; see evidence for detail.")
        )

        return ExplainabilityReport(
            explanation_text=explanation_text,
            evidence=items,
            confidence_calibration=min(overall_confidence, 1.0),
            provenance={
                "source": "TrustEvidence[B1,MBD,CP]",
                "source_layers": [e.source_layer for e in evidence],
                "layer_count": len(evidence),
            },
            validation_valid=overall_passed,
            validation_score=overall_score,
        )

    def explain(self, validation_assessment: Dict[str, Any]) -> ExplainabilityReport:
        if validation_assessment is None:
            raise ValueError(
                "ExplainabilityEngine.explain: validation_assessment is required "
                "(B2 has no other permitted input)."
            )

        valid = bool(validation_assessment.get("valid", True))
        fatal = bool(validation_assessment.get("fatal", False))
        score = float(validation_assessment.get("score", 1.0))
        confidence = float(validation_assessment.get("confidence", 1.0))
        reasons: List[str] = list(validation_assessment.get("reasons") or [])
        checks: Dict[str, Any] = dict(validation_assessment.get("checks") or {})
        details: Dict[str, Any] = dict(validation_assessment.get("details") or {})

        evidence = self._build_evidence(valid, checks, details, reasons)
        explanation_text = self._build_narrative(valid, fatal, score, reasons, evidence)
        calibration = self._calibrate_confidence(confidence, evidence)

        return ExplainabilityReport(
            explanation_text=explanation_text,
            evidence=evidence,
            confidence_calibration=calibration,
            provenance={
                "source": "B1.ValidationAssessment",
                "checks_seen": list(checks.keys()),
                "details_seen": list(details.keys()),
                "reason_count": len(reasons),
            },
            validation_valid=valid,
            validation_score=score,
        )

    # -- internal helpers ---------------------------------------------

    def _build_evidence(
        self,
        valid: bool,
        checks: Dict[str, Any],
        details: Dict[str, Any],
        reasons: List[str],
    ) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []

        # One evidence item per explicit B1 check result, if present.
        for check_name, check_value in checks.items():
            passed = bool(check_value) if not isinstance(check_value, dict) else bool(
                check_value.get("passed", check_value.get("valid", True))
            )
            evidence.append(
                EvidenceItem(
                    factor=check_name,
                    description=self.config.check_descriptions.get(
                        check_name, f"B1 check '{check_name}' result."
                    ),
                    weight=1.0 / max(len(checks), 1),
                    direction=(
                        EvidenceDirection.SUPPORTS
                        if passed
                        else EvidenceDirection.CONTRADICTS
                    ),
                    source_field=f"checks.{check_name}",
                )
            )

        # Fall back to reason strings when no structured checks exist.
        if not evidence and reasons:
            weight = 1.0 / max(len(reasons), 1)
            for reason in reasons:
                evidence.append(
                    EvidenceItem(
                        factor=reason,
                        description=f"B1 flagged: {reason}.",
                        weight=weight,
                        direction=EvidenceDirection.CONTRADICTS,
                        source_field="reasons",
                    )
                )

        if not evidence:
            evidence.append(
                EvidenceItem(
                    factor="no_anomalies",
                    description="No B1 checks or reasons flagged an anomaly.",
                    weight=1.0,
                    direction=(
                        EvidenceDirection.SUPPORTS if valid else EvidenceDirection.NEUTRAL
                    ),
                    source_field=None,
                )
            )

        return evidence

    def _build_narrative(
        self,
        valid: bool,
        fatal: bool,
        score: float,
        reasons: List[str],
        evidence: List[EvidenceItem],
    ) -> str:
        if fatal:
            cause = ", ".join(reasons) if reasons else "an unspecified fatal condition"
            return (
                f"B1 rejected this message with a fatal validation failure "
                f"(score={score:.2f}). Contributing factors: {cause}."
            )
        if not valid:
            cause = ", ".join(reasons) if reasons else "one or more validation checks"
            return (
                f"B1 marked this message invalid (score={score:.2f}) due to {cause}. "
                f"{len(evidence)} evidence factor(s) were considered."
            )
        return (
            f"B1 validated this message successfully (score={score:.2f}). "
            f"{len(evidence)} check(s)/factor(s) were consistent with a valid message."
        )

    def _calibrate_confidence(
        self, base_confidence: float, evidence: List[EvidenceItem]
    ) -> float:
        """Confidence in the *explanation itself*, not a re-score of B1.

        Lower when evidence is sparse (few factors to explain the verdict
        from), unchanged otherwise.
        """
        if not evidence:
            return min(base_confidence, self.config.sparse_evidence_confidence_cap)
        if len(evidence) == 1 and evidence[0].factor == "no_anomalies":
            return base_confidence
        return base_confidence
