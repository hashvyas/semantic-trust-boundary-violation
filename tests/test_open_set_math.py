"""
tests/test_open_set_math.py
==============================
Unit tests for the statistical machinery in b3_eval/run_open_set_analysis.py.
Runs WITHOUT the checkpoint or torch -- these are the parts of the open-set
determination that can be verified here, and a decision hangs on them being
correct (an inverted score convention would flip the recommendation, which
is exactly the bug this test was written to catch and did catch).

Run with:  python3 tests/test_open_set_math.py
"""
from __future__ import annotations

import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from b3_eval.run_open_set_analysis import auroc, energy, risk_coverage, softmax2

_FAILURES = []


def check(name, cond, evidence=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {evidence}" if evidence else ""))
    if not cond:
        _FAILURES.append(name)


def main():
    print("=" * 78)
    print("OPEN-SET ANALYSIS MATH")
    print("=" * 78)

    # --- AUROC ---------------------------------------------------------------
    print("\n--- AUROC (positive class must score HIGH) ---")
    check("Perfect separation -> 1.0", abs(auroc([3, 4, 5], [0, 1, 2]) - 1.0) < 1e-9)
    check("Perfectly reversed -> 0.0", abs(auroc([0, 1, 2], [3, 4, 5]) - 0.0) < 1e-9)
    check("Identical distributions (all ties) -> 0.5",
          abs(auroc([1, 1, 1], [1, 1, 1]) - 0.5) < 1e-9)
    check("Handles ties without bias", abs(auroc([1, 2], [1, 2]) - 0.5) < 1e-9)

    # --- risk-coverage / AURC ------------------------------------------------
    print("\n--- Risk-coverage / AURC (lower AURC is better) ---")
    _, aurc_perfect = risk_coverage([0.9, 0.8, 0.7], [1, 1, 1])
    check("Perfect model -> AURC 0", abs(aurc_perfect) < 1e-9)
    _, aurc_worst = risk_coverage([0.9, 0.8], [0, 0])
    check("All-wrong model -> AURC 1", abs(aurc_worst - 1.0) < 1e-9)
    _, aurc_good = risk_coverage([0.9, 0.8, 0.2], [1, 1, 0])   # error is LEAST confident
    _, aurc_bad = risk_coverage([0.9, 0.8, 0.2], [0, 1, 1])    # error is MOST confident
    check("Rewards confidence that ranks errors last",
          aurc_good < aurc_bad, f"AURC {aurc_good:.3f} (good ranking) < {aurc_bad:.3f} (bad)")
    check("Risk is monotone non-decreasing as coverage grows on a well-ranked model",
          all(b >= a - 1e-9 for (_, a), (_, b) in
              zip(risk_coverage([0.9, 0.8, 0.7, 0.2], [1, 1, 1, 0])[0][:-1],
                  risk_coverage([0.9, 0.8, 0.7, 0.2], [1, 1, 1, 0])[0][1:])))

    # --- energy score sign convention (the bug this test caught) -------------
    print("\n--- Energy score: HIGHER must mean MORE OOD ---")
    e_confident = energy(0.0, 8.0)    # large logit gap = confident, in-distribution
    e_flat = energy(0.0, 0.0)         # flat logits = uncertain / OOD-like
    e_small = energy(0.1, 0.2)        # small-magnitude logits = OOD-like
    check("Confident (ID-like) input scores LOWER energy than flat input",
          e_confident < e_flat, f"confident={e_confident:.3f} < flat={e_flat:.3f}")
    check("Confident input scores LOWER energy than small-magnitude logits",
          e_confident < e_small, f"confident={e_confident:.3f} < small={e_small:.3f}")
    check("Energy is monotone: larger logit magnitude -> lower (more ID) energy",
          energy(0.0, 12.0) < energy(0.0, 8.0) < energy(0.0, 4.0))

    # End-to-end: energy must give AUROC > 0.5 separating OOD (flat) from ID
    # (confident). This is the exact composition used in the real harness -- an
    # inverted sign here would silently invert the reported AUROC.
    rng = random.Random(0)
    id_logits = [(0.0, rng.uniform(6.0, 10.0)) for _ in range(200)]      # confident ID
    ood_logits = [(0.0, rng.uniform(-0.5, 0.5)) for _ in range(200)]     # flat OOD
    id_e = [energy(a, b) for a, b in id_logits]
    ood_e = [energy(a, b) for a, b in ood_logits]
    au = auroc(ood_e, id_e)   # OOD is the positive class, must score HIGH
    check("End-to-end: energy separates OOD from ID with AUROC >> 0.5 (correct sign)",
          au > 0.95, f"AUROC={au:.3f} (an inverted sign would give ~{1-au:.3f})")

    # --- softmax -------------------------------------------------------------
    print("\n--- softmax2 ---")
    p0, p1 = softmax2(0.0, 0.0)
    check("Equal logits -> (0.5, 0.5)", abs(p0 - 0.5) < 1e-9 and abs(p1 - 0.5) < 1e-9)
    check("Normalized", abs(sum(softmax2(2.0, -3.0)) - 1.0) < 1e-9)
    check("Temperature T>1 softens (moves p toward 0.5)",
          softmax2(0.0, 4.0, T=1.0)[1] > softmax2(0.0, 4.0, T=4.0)[1])

    print()
    print("=" * 78)
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S): {_FAILURES}")
        sys.exit(1)
    print("All open-set analysis math checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
