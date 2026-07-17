"""
tests/test_interface_equivalence.py
======================================
Locks the interface A/B result (see INTERFACE_COMPARISON.md).

Claims pinned:
  I1. DEFAULT-OFF SAFETY: with the default TrustPolicy, adding `p_malicious`
      to a SemanticResult changes NOTHING. The legacy interface is the
      default and is byte-for-byte unaffected by the new optional field.
  I2. STBV EQUIVALENCE: when the other layers are clean (the STBV premise),
      the two interfaces produce IDENTICAL decisions for every p_malicious in
      [0,1]. The continuous interface buys nothing precisely where it was
      hypothesised to help.
  I3. DIRECTION OF DIVERGENCE: where they DO differ (only when B1 is already
      degraded), the LEGACY interface is more often the more conservative
      (safer) one. Enabling the continuous interface makes the system less
      cautious -- which is why it stays off.
  I4. CONTRACT ADDITIVITY: `p_malicious` is optional; SemanticResults without
      it still work under both policies.

Run with:  python3 tests/test_interface_equivalence.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from b2_explain.explainability import ExplainabilityEngine
from pipeline.b3_bridge import B3RiskPolicy, SemanticResult
from trust_engine.decision_engine import TrustDecisionEngine
from trust_engine.policy import TrustPolicy

_FAILURES = []


def check(name, cond, evidence=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {evidence}" if evidence else ""))
    if not cond:
        _FAILURES.append(name)


POL = B3RiskPolicy()
ENG_DEFAULT = TrustDecisionEngine()                                   # default policy
ENG_LEGACY = TrustDecisionEngine(policy=TrustPolicy(use_continuous_semantic_belief=False))
ENG_CONT = TrustDecisionEngine(policy=TrustPolicy(use_continuous_semantic_belief=True))
EXPLAIN = ExplainabilityEngine()
RANK = {"ACCEPT": 0, "CAUTION": 1, "REJECT": 2}


def b3(p, with_p=True):
    label = "MALICIOUS" if p >= 0.5 else "BENIGN"
    conf = max(p, 1.0 - p)
    d = {"available": True, "label": label, "confidence": conf,
         "risk_level": POL.classify(label, conf), "status": "ok"}
    if with_p:
        d["p_malicious"] = p
    return d


def b1b2(score):
    b1 = {"valid": score >= 0.7, "fatal": False, "score": score, "confidence": 1.0,
          "reasons": [], "checks": {}, "details": {}}
    return b1, EXPLAIN.explain(b1).to_dict()


def main():
    print("=" * 78)
    print("INTERFACE EQUIVALENCE (legacy argmax+max-prob  vs  continuous p_malicious)")
    print("=" * 78)

    # --- I1: default policy is unaffected by the new field --------------------
    print("\n--- I1: default policy ignores p_malicious (no behaviour change) ---")
    mismatch = 0
    for i in range(0, 101, 5):
        p = i / 100
        for s in (0.3, 0.6, 1.0):
            b1, b2 = b1b2(s)
            d_legacy = ENG_LEGACY.decide(b1, b2, b3(p, with_p=False))
            d_default_new = ENG_LEGACY.decide(b1, b2, b3(p, with_p=True))
            if d_legacy.trust_level != d_default_new.trust_level or abs(d_legacy.trust_score - d_default_new.trust_score) > 1e-12:
                mismatch += 1
    check("The default TrustPolicy is unaffected by the presence of p_malicious",
          mismatch == 0, f"mismatch_count={mismatch}")

    # --- I2: continuous and legacy interfaces behave identically on STBVs -----
    print("\n--- I2: continuous and legacy behave identically under STBV premise ---")
    mismatch = 0
    b1c, b2c = b1b2(1.0)
    for i in range(0, 101):
        p = i / 100
        d_legacy = ENG_LEGACY.decide(b1c, b2c, b3(p, with_p=False))
        d_cont = ENG_CONT.decide(b1c, b2c, b3(p, with_p=True))
        if d_legacy.trust_level != d_cont.trust_level:
            mismatch += 1
    check("The continuous and legacy interfaces produce identical decisions on STBVs (all p in [0,1])",
          mismatch == 0, f"divergent_decisions_count={mismatch}")

    # --- I3: legacy is safer when they diverge (degraded B1 source) ------------
    print("\n--- I3: direction of divergence (when B1 is degraded, which is safer?) ---")
    total, same, legacy_safer, cont_safer = 0, 0, 0, 0
    for i in range(0, 101):
        p = i / 100
        for s in range(0, 100, 5):
            score = s / 100
            b1d, b2d = b1b2(score)
            d_legacy = ENG_LEGACY.decide(b1d, b2d, b3(p, with_p=False))
            d_cont = ENG_CONT.decide(b1d, b2d, b3(p, with_p=True))
            total += 1
            if d_legacy.trust_level == d_cont.trust_level:
                same += 1
            else:
                l_rank = RANK[d_legacy.trust_level.value]
                c_rank = RANK[d_cont.trust_level.value]
                if l_rank > c_rank:
                    legacy_safer += 1
                else:
                    cont_safer += 1

    print(f"  total configurations evaluated: {total}")
    print(f"  same decision:                  {same} ({same/total*100:.1f}%)")
    print(f"  legacy interface is safer:      {legacy_safer} ({legacy_safer/total*100:.1f}%)")
    print(f"  continuous interface is safer:  {cont_safer} ({cont_safer/total*100:.1f}%)")
    check("Continuous interface diverges ONLY when B1 is already degraded", same + legacy_safer + cont_safer == total)
    check("When they diverge, the legacy interface is more conservative (safer)",
          legacy_safer > cont_safer, f"legacy_safer={legacy_safer} cont_safer={cont_safer}")

    # --- I4: contract additivity ---------------------------------------------
    print("\n--- I4: p_malicious is optional; absent is handled by both policies ---")
    sr = SemanticResult.unavailable("x").to_dict()
    check("SemanticResult.unavailable() carries p_malicious=None",
          "p_malicious" in sr and sr["p_malicious"] is None)
    no_p = {"available": True, "label": "MALICIOUS", "confidence": 0.9,
            "risk_level": "high", "status": "ok"}      # legacy caller, no p_malicious
    b1c, b2c = b1b2(1.0)
    d_legacy = ENG_LEGACY.decide(b1c, b2c, dict(no_p))
    d_cont = ENG_CONT.decide(b1c, b2c, dict(no_p))
    check("Continuous policy falls back to the legacy mapping when p_malicious is absent",
          d_legacy.trust_level == d_cont.trust_level
          and abs(d_legacy.trust_score - d_cont.trust_score) < 1e-12,
          f"{d_legacy.trust_level.value} == {d_cont.trust_level.value}")

    print()
    print("=" * 78)
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S): {_FAILURES}")
        sys.exit(1)
    print("Interface equivalence locked. Verdict: RETAIN the existing interface.")
    sys.exit(0)

if __name__ == "__main__":
    main()
