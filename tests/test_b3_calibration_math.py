"""
tests/test_b3_calibration_math.py
====================================
Unit test for b3_eval/run_calibration.py's temperature-scaling and ECE/Brier
implementations. Runs WITHOUT the B3 checkpoint or torch -- it validates the
calibration mathematics against synthetic logits with a KNOWN injected
temperature, which is the only part of Part 7 that can be verified without
GPU hardware.

Run with:  python3 tests/test_b3_calibration_math.py
"""
from __future__ import annotations

import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from b3_eval.run_calibration import _softmax2, ece_and_brier, fit_temperature

_FAILURES = []


def check(name, cond, evidence=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {evidence}" if evidence else ""))
    if not cond:
        _FAILURES.append(name)


def make_data(inflation, n=2000, seed=0):
    """Model reports logits inflated by `inflation`x relative to the true
    log-odds; labels are SAMPLED from the true probability (so the model is
    genuinely overconfident, not merely sharp-and-correct)."""
    rng = random.Random(seed)
    logits, labels = [], []
    for _ in range(n):
        true_p = rng.uniform(0.05, 0.95)
        y = 1 if rng.random() < true_p else 0
        z = math.log(true_p / (1 - true_p))
        logits.append((0.0, z * inflation))
        labels.append(y)
    return logits, labels


def metrics(logits, labels, T):
    confs, correct, pos = [], [], []
    for (z0, z1), y in zip(logits, labels):
        p0, p1 = _softmax2(z0 / T, z1 / T)
        confs.append(max(p0, p1))
        correct.append(int((1 if p1 >= p0 else 0) == y))
        pos.append(p1)
    ece, brier, _ = ece_and_brier(confs, correct, pos, labels)
    return ece, brier


def main():
    print("=" * 78)
    print("B3 CALIBRATION MATH (no checkpoint / no GPU required)")
    print("=" * 78)

    # --- softmax sanity ---
    p0, p1 = _softmax2(0.0, 0.0)
    check("softmax2 of equal logits is (0.5, 0.5)", abs(p0 - 0.5) < 1e-9 and abs(p1 - 0.5) < 1e-9)
    p0, p1 = _softmax2(-10.0, 10.0)
    check("softmax2 is monotone and normalized", p1 > 0.99 and abs(p0 + p1 - 1.0) < 1e-9)

    # --- temperature recovery on a 3x-overconfident model ---
    logits, labels = make_data(inflation=3.0)
    ece0, brier0 = metrics(logits, labels, 1.0)
    T = fit_temperature(logits, labels)
    ece1, brier1 = metrics(logits, labels, T)
    print(f"\n  3x-overconfident model: ECE {ece0:.4f} -> {ece1:.4f}, fitted T={T:.3f}")
    check("Temperature recovers expected inflation (T is near 3.0)", abs(T - 3.0) < 0.25, f"T={T:.3f}")
    check("ECE decreases after temperature-scaling", ece1 < ece0, f"{ece0:.4f} -> {ece1:.4f}")
    check("Brier score decreases after temperature-scaling", brier1 < brier0, f"{brier0:.4f} -> {brier1:.4f}")

    # --- calibration on a near-calibrated model ---
    logits_c, labels_c = make_data(inflation=1.05, seed=1)
    ece_c0, _ = metrics(logits_c, labels_c, 1.0)
    T_c = fit_temperature(logits_c, labels_c)
    ece_c1, _ = metrics(logits_c, labels_c, T_c)
    print(f"\n  Near-calibrated model: ECE {ece_c0:.4f} -> {ece_c1:.4f}, fitted T={T_c:.3f}")
    check("Fitted temperature is near 1.0", abs(T_c - 1.0) < 0.15, f"T={T_c:.3f}")
    check("ECE remains low", ece_c1 <= ece_c0 + 0.02, f"{ece_c0:.4f} -> {ece_c1:.4f}")

    # --- underconfident model: T should shrink below 1 ---
    logits_u, labels_u = make_data(inflation=0.4, seed=3)
    T_u = fit_temperature(logits_u, labels_u)
    check("Underconfident model fits T < 1.0 (sharpens)", T_u < 1.0, f"T={T_u:.3f}")

    print()
    print("=" * 78)
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S): {_FAILURES}")
        sys.exit(1)
    print("All B3 calibration math checks passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
