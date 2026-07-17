"""
tests/test_abstain_semantics.py
==================================
Regression-locks the UNKNOWN/ABSTAIN determination (see
UNKNOWN_ABSTAIN_DETERMINATION.md). These are not aspirational tests -- they
pin the *current, measured* behaviour of the fusion path so the determination's
evidence cannot silently rot.

Three claims are locked:

  C1. B3 already performs graded soft-abstention. A low-confidence MALICIOUS
      verdict yields high Theta (ignorance) mass and lands in CAUTION, not
      REJECT. CAUTION *is* the abstain state; it is produced by the Trust
      Decision Engine, which is the layer that owns decisions. No new B3
      class is needed to obtain this behaviour.

  C2. A discrete UNKNOWN label mapped to VACUOUS mass is a provable no-op for
      safety. Under the STBV premise (PKI/B1/MBD/CP all clean -- that is what
      makes it an STBV), a vacuous semantic mass yields exactly the same
      decision (ACCEPT) as a confidently-WRONG BENIGN verdict. Ignorance mass
      by DS construction cannot move a verdict away from what the other
      layers concluded (Shafer 1976). Therefore UNKNOWN-as-vacuous adds
      nothing.

  C3. The real gap is not a missing UNKNOWN class but an information loss in
      the argmax->max-prob->band contract: all semantic suspicion below
      p_malicious = 0.5 is discarded. p_mal=0.49 (near coin-flip on an
      attack) is indistinguishable from p_mal=0.05. This test pins that
      dead zone so any future fix is visible as a diff.

Run with:  python3 tests/test_abstain_semantics.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from b2_explain.explainability import ExplainabilityEngine
from pipeline.b3_bridge import B3RiskPolicy
from trust_engine.decision_engine import TrustDecisionEngine

_FAILURES = []


def check(name, cond, evidence=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {evidence}" if evidence else ""))
    if not cond:
        _FAILURES.append(name)


TE = TrustDecisionEngine()
POL = B3RiskPolicy()
B1_CLEAN = {"valid": True, "fatal": False, "score": 1.0, "confidence": 1.0,
            "reasons": [], "checks": {}, "details": {}}
B2_CLEAN = ExplainabilityEngine().explain(B1_CLEAN).to_dict()


def decide(b3):
    return TE.decide(B1_CLEAN, B2_CLEAN, b3)


def b3_from_softmax(p_mal: float):
    """Exactly what inference.py emits: argmax label + max-prob confidence."""
    label = "MALICIOUS" if p_mal >= 0.5 else "BENIGN"
    conf = max(p_mal, 1.0 - p_mal)
    return {"available": True, "label": label, "confidence": conf,
            "risk_level": POL.classify(label, conf), "status": "ok"}


def main():
    print("=" * 78)
    print("UNKNOWN / ABSTAIN SEMANTICS (regression lock)")
    print("=" * 78)

    # --- C1: graded soft-abstention already exists -----------------------------
    print("\n--- C1: low-confidence MALICIOUS already abstains (Theta mass -> CAUTION) ---")
    fd_low = decide(b3_from_softmax(0.55))
    fd_high = decide(b3_from_softmax(0.95))
    m_low = fd_low.details["ds_semantic_mass"]
    m_high = fd_high.details["ds_semantic_mass"]
    check("Low-confidence MALICIOUS -> CAUTION (soft abstain), not REJECT",
          fd_low.trust_level.value == "CAUTION", f"got={fd_low.trust_level.value}")
    check("High-confidence MALICIOUS -> REJECT",
          fd_high.trust_level.value == "REJECT", f"got={fd_high.trust_level.value}")
    check("Theta (ignorance) mass is monotonically higher when B3 is less confident",
          m_low["m_theta"] > m_high["m_theta"],
          f"m_theta {m_low['m_theta']:.2f} (conf .55) vs {m_high['m_theta']:.2f} (conf .95)")

    # --- C2: UNKNOWN-as-vacuous is a provable no-op ---------------------------
    print("\n--- C2: UNKNOWN mapped to vacuous mass is a NO-OP under the STBV premise ---")
    b3_ood_confidently_wrong = {"available": True, "label": "BENIGN", "confidence": 0.97,
                                 "risk_level": "none", "status": "ok"}
    b3_unknown_vacuous = {"available": False, "label": None, "confidence": None,
                           "risk_level": "unavailable", "status": "UNKNOWN/abstain"}
    d_wrong = decide(b3_ood_confidently_wrong)
    d_unknown = decide(b3_unknown_vacuous)
    check("Confidently-WRONG BENIGN on an OOD attack -> ACCEPT (the open-set failure)",
          d_wrong.trust_level.value == "ACCEPT", f"got={d_wrong.trust_level.value}")
    check("UNKNOWN-as-vacuous-mass -> ACCEPT as well (adds NO safety)",
          d_unknown.trust_level.value == "ACCEPT", f"got={d_unknown.trust_level.value}")
    check("C2 PROVEN: UNKNOWN-as-vacuous yields the SAME decision as a confidently-wrong BENIGN",
          d_wrong.trust_level == d_unknown.trust_level,
          "an UNKNOWN class mapped to ignorance mass cannot rescue an STBV attack")
    check("Vacuous semantic mass is genuinely all-ignorance (m_theta == 1.0)",
          abs(d_unknown.details["ds_semantic_mass"]["m_theta"] - 1.0) < 1e-9)

    # --- C3: the sub-0.5 dead zone --------------------------------------------
    print("\n--- C3: argmax->max-prob contract discards all suspicion below p_mal=0.5 ---")
    d_05 = decide(b3_from_softmax(0.05))
    d_49 = decide(b3_from_softmax(0.49))
    d_51 = decide(b3_from_softmax(0.51))
    m_05 = d_05.details["ds_semantic_mass"]
    m_49 = d_49.details["ds_semantic_mass"]
    check("p_mal=0.49 (near coin-flip on an attack) -> ACCEPT",
          d_49.trust_level.value == "ACCEPT", f"got={d_49.trust_level.value}")
    check("p_mal=0.05 -> ACCEPT (as it should)",
          d_05.trust_level.value == "ACCEPT")
    check("DEAD ZONE: p_mal=0.49 commits ZERO disbelief mass, exactly like p_mal=0.05",
          abs(m_49["m_not_A"]) < 1e-9 and abs(m_05["m_not_A"]) < 1e-9,
          f"m_not_A: p=.49 -> {m_49['m_not_A']:.2f}, p=.05 -> {m_05['m_not_A']:.2f}")
    check("Discontinuity at 0.5: p_mal=0.51 jumps straight to CAUTION",
          d_51.trust_level.value == "CAUTION", f"got={d_51.trust_level.value}")
    check("...so the decision is DISCONTINUOUS across an infinitesimal change in p_mal",
          d_49.trust_level != d_51.trust_level,
          "p=.49 -> ACCEPT but p=.51 -> CAUTION")

    print()
    print("=" * 78)
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S): {_FAILURES}")
        sys.exit(1)
    print("Abstain-semantics regression lock: all claims verified.")
    print("Determination: a discrete UNKNOWN class is NOT justified (C2).")
    print("The measurable gap is the sub-0.5 dead zone (C3) -- see")
    print("UNKNOWN_ABSTAIN_DETERMINATION.md and b3_eval/run_open_set_analysis.py.")
    sys.exit(0)

if __name__ == "__main__":
    main()
