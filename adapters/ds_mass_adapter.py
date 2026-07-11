"""
adapters/ds_mass_adapter.py
=============================
Converts a FinalTrustDecision into a Dempster-Shafer mass-function triple
(m_A, m_not_A, m_Theta), matching the belief/disbelief/uncertainty
convention already used by b2_csia.uncertainty.MassFunction elsewhere in
this repository (focal elements: 'A' = benign/trust, 'not_A' =
suspicious/distrust, 'Theta' = uncertain/ignorance). This is pure format
conversion -- no trust logic. All trust reasoning already happened in
TrustDecisionEngine; this adapter only re-expresses its already-computed
trust_score/confidence as a normalized DS mass assignment for downstream
consumers (simulation/fusion frameworks) that expect that representation.

Mapping (by construction, always sums to exactly 1.0):
    m_A      = trust_score * confidence       (belief in "benign")
    m_not_A  = (1 - trust_score) * confidence (belief in "suspicious")
    m_Theta  = 1 - confidence                 (ignorance / unresolved mass)

Rationale: confidence scales how much mass is committed to a verdict at
all vs. left as ignorance; trust_score splits the committed mass between
"benign" and "suspicious". This mirrors b2_csia's own
MassFunction.from_trust_confidence() convention so it composes cleanly
with any existing Dempster-Shafer fusion code downstream (e.g. DS MASS)
without requiring that code to change its input contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from adapters.base import Adapter
from trust_engine.models import FinalTrustDecision


@dataclass(frozen=True)
class DSMassOutput:
    """A normalized Dempster-Shafer mass-function triple. m_A + m_not_A +
    m_Theta always sums to 1.0 by construction (see module docstring)."""

    m_A: float       # belief: benign / trustworthy
    m_not_A: float   # belief: suspicious / not trustworthy
    m_Theta: float   # ignorance / uncertainty
    origin_module: str = "trust_engine"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "m_A": self.m_A,
            "m_not_A": self.m_not_A,
            "m_Theta": self.m_Theta,
            "origin_module": self.origin_module,
        }


class DSMassAdapter(Adapter):
    """Formats a FinalTrustDecision as a DS-MASS-compatible mass function."""

    def __init__(self, origin_module: str = "trust_engine") -> None:
        self.origin_module = origin_module

    def adapt(self, decision: FinalTrustDecision) -> DSMassOutput:
        trust = max(0.0, min(1.0, decision.trust_score))
        confidence = max(0.0, min(1.0, decision.confidence))

        m_a = trust * confidence
        m_not_a = (1.0 - trust) * confidence
        m_theta = 1.0 - confidence

        return DSMassOutput(
            m_A=m_a,
            m_not_A=m_not_a,
            m_Theta=m_theta,
            origin_module=self.origin_module,
        )
