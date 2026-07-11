"""
contracts/trust_evidence.py
===============================
TrustEvidence: the common lens B2 reads across B1/MBD/CP, per the
approved architecture (frozen). This is a VIEW, not a replacement --
ValidationAssessment/MBDResult/CPResult keep their own rich,
layer-specific shapes; TrustEvidence is a thin uniform wrapper each
producer supplies alongside its native result so B2 doesn't need
per-layer branching logic to explain three different dict shapes.

Lives in a neutral `contracts/` package (not trust_engine/, not
b2_explain/) deliberately: B2 needs to consume TrustEvidence, and B2 is
architecturally forbidden from importing trust_engine (see
tests/verify_dependency_graph.py). A shared, zero-dependency contracts
package is how both B2 and the Trust Decision Engine can agree on this
shape without either importing the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class TrustEvidence:
    """Common evidence lens. One instance per upstream producer
    (B1, MBD, CP) per message."""

    source_layer: str          # "B1_VALIDATION" | "MBD" | "CP"
    passed: bool
    score: float                # normalized [0,1], layer-specific meaning
    confidence: float
    findings: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)  # original ValidationAssessment/MBDResult/CPResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_layer": self.source_layer,
            "passed": self.passed,
            "score": self.score,
            "confidence": self.confidence,
            "findings": list(self.findings),
            "raw": self.raw,
        }

    @staticmethod
    def from_validation_assessment(va: Dict[str, Any]) -> "TrustEvidence":
        """Wraps B1's ValidationAssessment dict."""
        return TrustEvidence(
            source_layer="B1_VALIDATION",
            passed=bool(va.get("valid", True)),
            score=float(va.get("score", 1.0)),
            confidence=float(va.get("confidence", 1.0)),
            findings=list(va.get("reasons") or []),
            raw=va,
        )

    @staticmethod
    def from_mbd_result(mbd: Dict[str, Any]) -> "TrustEvidence":
        """Wraps MBD's MBDResult dict. score = 1 - anomaly_score (MBD
        reports anomaly, TrustEvidence's score convention is
        trust-direction, i.e. higher = more trustworthy)."""
        anomaly = float(mbd.get("anomaly_score", 0.0))
        return TrustEvidence(
            source_layer="MBD",
            passed=bool(mbd.get("passed", mbd.get("mbd_pass", True))),
            score=max(0.0, 1.0 - anomaly),
            confidence=float(mbd.get("behavior_evidence_quality", 0.5)),
            findings=list(mbd.get("evidence") or []),
            raw=dict(mbd),
        )

    @staticmethod
    def from_cp_result(cp: Dict[str, Any]) -> "TrustEvidence":
        """Wraps CP's fusion-result dict."""
        findings = []
        if cp.get("num_reports", 0) > 0:
            findings.append(
                f"CP fused {cp.get('num_reports')} report(s) from "
                f"{len(cp.get('senders', []))} sender(s); "
                f"fusion_confidence={cp.get('fusion_confidence')}."
            )
        return TrustEvidence(
            source_layer="CP",
            passed=bool(cp.get("cp_pass", False)),
            score=float(cp.get("fusion_confidence", cp.get("cp_confidence", 0.0))),
            confidence=float(cp.get("fusion_confidence", cp.get("cp_confidence", 0.0))),
            findings=findings,
            raw=cp,
        )
