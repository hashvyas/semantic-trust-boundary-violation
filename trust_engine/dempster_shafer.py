"""
trust_engine/dempster_shafer.py
==================================
Real Dempster-Shafer evidence combination for the Trust Decision Engine.
Implements roadmap item A3/B1: "Implement and formalize the fusion rule"
-- replaces the previous rank-based ("most conservative of the two wins")
placeholder with an actual, citable Dempster's combination rule over a
shared belief/disbelief/uncertainty frame of discernment.

Frame of discernment: Theta = {A, not_A}, where A = "trustworthy /
benign" and not_A = "suspicious / malicious". A mass function m assigns
belief mass across the focal elements {A}, {not_A}, {A, not_A} (written
here as m_A, m_not_A, m_theta), with m_A + m_not_A + m_theta == 1.

This reuses the exact belief/disbelief/uncertainty convention already
established in this repository by adapters/ds_mass_adapter.py and
b2_csia.uncertainty.MassFunction (see those modules' docstrings), so the
mapping from a (score, confidence) pair to a mass triple is consistent
everywhere in the codebase, not a one-off invention for this module:

    m_A     = score * confidence           (mass committed to "benign")
    m_not_A = (1 - score) * confidence     (mass committed to "suspicious")
    m_theta = 1 - confidence               (unresolved / ignorance mass)

Dempster's combination rule (Dempster 1967; Shafer 1976) for two
independent mass functions m1, m2 over the same frame:

    K = sum over disjoint focal-element pairs (B1 in m1, B2 in m2) with
        B1 ∩ B2 = ∅, of m1(B1) * m2(B2)
                                                              (conflict mass)

    m12(C) = ( sum over B1 ∩ B2 = C of m1(B1) * m2(B2) ) / (1 - K)

KNOWN LIMITATION OF RAW DEMPSTER NORMALIZATION (Zadeh 1984): dividing by
(1-K) discards the conflict mass entirely, which produces
counterintuitive results whenever two sources are both highly
confident AND disagree -- the classic textbook example (Zadeh's
counterexample) is two doctors, each 99% sure of a different diagnosis,
whose naive Dempster combination becomes near-100% confident of
whichever diagnosis has the marginally larger product term, rather than
reporting "these two confident opinions flatly disagree." Verified
directly in this codebase (see git history / tests/
test_dempster_shafer_fusion.py's "KNOWN LIMITATION" tests): combining a
clean, highly-confident B1/B2 crypto source with a highly-confident B3
MALICIOUS verdict under raw normalization pushed the combined result
TOWARD trustworthy, not away from it -- exactly backwards for a
security system, where a confident attack signal conflicting with a
confident "looks clean" signal should raise alarm, not suppress it.

`combine()` therefore implements Yager's modified combination rule
(Yager, 1987, "On the Dempster-Shafer framework and new combination
rules", Information Sciences 41(2)) instead of raw normalization:
conflict mass K is reassigned to ignorance (Theta) rather than divided
out:

    K         = m1_A * m2_not_A + m1_not_A * m2_A
    m12_A     = m1_A*m2_A + m1_A*m2_theta + m1_theta*m2_A
    m12_not_A = m1_not_A*m2_not_A + m1_not_A*m2_theta + m1_theta*m2_not_A
    m12_theta = m1_theta*m2_theta + K

(m12_A + m12_not_A + m12_theta == 1 by construction, no renormalization
needed or performed.) This correctly reports strong, confident
disagreement as high uncertainty rather than silently discarding it,
while reducing to the same result as raw Dempster combination whenever
K is small (the two sources are not in serious conflict) -- the
common, low-conflict case.

KNOWN LIMITATION, HANDLED DELIBERATELY (not silently): even Yager's
rule, on its own, will not push a decision all the way to REJECT purely
from combining one confident "clean" crypto source against one
confident "malicious" semantic source -- it correctly reports the
disagreement as elevated uncertainty (CAUTION-range), not as a
decisive verdict either way, which is the mathematically honest
outcome for two conflicting opinions with no further evidence to break
the tie. TrustDecisionEngine.decide() therefore applies two additional,
explicitly-documented policy floors on top of the fused score for the
two cases where this system's threat model demands a decisive,
asymmetric-cost response regardless of what a symmetric evidence
combination alone would report: (1) B1 fatal (unchanged, pre-existing
rule) and (2) B3 HIGH-confidence semantic risk. See
trust_engine/decision_engine.py's decide() for exactly which floors are
applied and why -- they are policy decisions layered on top of the
fusion math, not asserted to be a property of Yager's rule itself.

The pignistic probability transform (Smets & Kennes, 1994) is used to
turn a combined mass function back into a single scalar trust score for
banding against the same thresholds already used elsewhere:

    trust_score(m) = m_A + 0.5 * m_theta
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MassFunction:
    """A Dempster-Shafer mass function over the 2-element frame {A, not_A}.

    m_A + m_not_A + m_theta must equal 1.0 (validated in __post_init__).
    """

    m_A: float
    m_not_A: float
    m_theta: float

    def __post_init__(self) -> None:
        total = self.m_A + self.m_not_A + self.m_theta
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"MassFunction masses must sum to 1.0, got {total:.6f} "
                              f"(m_A={self.m_A}, m_not_A={self.m_not_A}, m_theta={self.m_theta})")
        for name, v in (("m_A", self.m_A), ("m_not_A", self.m_not_A), ("m_theta", self.m_theta)):
            if not (0.0 <= v <= 1.0 + 1e-9):
                raise ValueError(f"MassFunction.{name} must be in [0, 1], got {v}")

    @staticmethod
    def from_score_confidence(score: float, confidence: float) -> "MassFunction":
        """Standard repo-wide convention (see module docstring) for turning
        a (score, confidence) pair into a mass triple. `score` is a
        trust-like value in [0, 1] (1.0 = fully trustworthy); `confidence`
        is how much mass the source is willing to commit at all (1.0 =
        fully committed, 0.0 = total ignorance)."""
        score = max(0.0, min(1.0, float(score)))
        confidence = max(0.0, min(1.0, float(confidence)))
        m_a = score * confidence
        m_not_a = (1.0 - score) * confidence
        m_theta = 1.0 - confidence
        # Guard against floating-point drift so __post_init__'s sum check
        # never spuriously fails.
        drift = 1.0 - (m_a + m_not_a + m_theta)
        m_theta += drift
        return MassFunction(m_A=m_a, m_not_A=m_not_a, m_theta=m_theta)

    @staticmethod
    def vacuous() -> "MassFunction":
        """Total ignorance: all mass on Theta. Used when a source has no
        opinion at all (e.g. B3 unavailable)."""
        return MassFunction(m_A=0.0, m_not_A=0.0, m_theta=1.0)

    def pignistic_trust_score(self) -> float:
        """Pignistic probability transform (Smets & Kennes 1994): splits
        ignorance mass evenly between A and not_A to produce a single
        scalar trust score in [0, 1] for threshold banding."""
        return self.m_A + 0.5 * self.m_theta

    def conflict_with(self, other: "MassFunction") -> float:
        """K: the mass assigned to the empty set when combining self with
        other -- i.e. how much the two sources directly contradict each
        other (one says A, the other says not_A, with nothing left as
        ignorance to absorb the disagreement)."""
        return self.m_A * other.m_not_A + self.m_not_A * other.m_A


def combine(m1: MassFunction, m2: MassFunction) -> MassFunction:
    """Yager's modified combination rule (Yager 1987) for two independent
    mass functions over the {A, not_A} frame -- see module docstring for
    the full derivation and the citable reason this codebase uses Yager's
    rule rather than raw Dempster normalization. Conflict mass K is
    reassigned to Theta (ignorance) instead of divided out, so the
    result is always well-defined (no division, no K==1 special case
    needed) and correctly represents high-confidence disagreement as
    high uncertainty rather than an arbitrary, discarded quantity.
    """
    K = m1.conflict_with(m2)
    combined_a = m1.m_A * m2.m_A + m1.m_A * m2.m_theta + m1.m_theta * m2.m_A
    combined_not_a = m1.m_not_A * m2.m_not_A + m1.m_not_A * m2.m_theta + m1.m_theta * m2.m_not_A
    combined_theta = m1.m_theta * m2.m_theta + K

    # Renormalize away floating-point drift only (should already sum to
    # 1.0 exactly by construction -- unlike raw Dempster combination,
    # Yager's rule needs no division).
    total = combined_a + combined_not_a + combined_theta
    if total > 0:
        combined_a /= total
        combined_not_a /= total
        combined_theta /= total

    return MassFunction(m_A=combined_a, m_not_A=combined_not_a, m_theta=combined_theta)
