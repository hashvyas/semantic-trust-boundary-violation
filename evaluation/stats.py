"""
evaluation/stats.py
======================
Part 10 (Statistical validation): bootstrap confidence intervals, McNemar's
test, Wilcoxon signed-rank, paired t-test, and effect sizes -- each with an
explicit applicability guard so tests are only applied where statistically
appropriate (per the mandate: "only where statistically appropriate").

Design rule: every function returns a dict that INCLUDES its own
applicability verdict and any warnings, so downstream table generators can
print "n too small for McNemar exact conditions" instead of a misleading
p-value.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence

try:
    from scipy import stats as _scipy_stats
    _SCIPY = True
except Exception:
    _SCIPY = False


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> Dict:
    """Percentile bootstrap CI for the mean of `values`. Appropriate for any
    metric computed per-scenario/per-seed where the sampling unit is
    independent runs (accuracy per seed, F1 per scenario)."""
    values = list(values)
    if len(values) < 2:
        return {"applicable": False, "reason": f"need >=2 samples, got {len(values)}"}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return {"applicable": True, "mean": sum(values) / len(values),
            "ci_low": lo, "ci_high": hi, "alpha": alpha, "n": len(values),
            "n_boot": n_boot, "warning": ("n < 10: CI is wide and unstable; "
                                            "report but interpret cautiously") if len(values) < 10 else None}


def mcnemar(preds_a: Sequence[bool], preds_b: Sequence[bool],
            truths: Sequence[bool]) -> Dict:
    """McNemar's test for paired classifier comparison on the SAME items.
    Appropriate when comparing configuration A vs B on identical inputs
    (e.g., B1+B2 vs full stack on the same message sequence). Uses the
    exact binomial form when discordant pairs < 25 (chi-square approx is
    invalid there)."""
    if not (len(preds_a) == len(preds_b) == len(truths)):
        return {"applicable": False, "reason": "unequal lengths (must be paired)"}
    b = sum(1 for pa, pb, t in zip(preds_a, preds_b, truths) if (pa == t) and (pb != t))
    c = sum(1 for pa, pb, t in zip(preds_a, preds_b, truths) if (pa != t) and (pb == t))
    n_discordant = b + c
    if n_discordant == 0:
        return {"applicable": False, "reason": "zero discordant pairs -- classifiers "
                "identical on this data; no test needed", "b": b, "c": c}
    if _SCIPY:
        if n_discordant < 25:
            p = float(_scipy_stats.binomtest(min(b, c), n_discordant, 0.5).pvalue)
            form = "exact binomial (discordant<25)"
        else:
            chi2 = (abs(b - c) - 1) ** 2 / n_discordant
            p = float(1 - _scipy_stats.chi2.cdf(chi2, df=1))
            form = "chi-square with continuity correction"
    else:
        # Fallback to pure Python implementation using math library
        if n_discordant < 25:
            k_min = min(b, c)
            # Two-sided binomial test p-value: sum of binomial probabilities
            p = 2.0 * sum(math.comb(n_discordant, k) for k in range(k_min + 1)) * (0.5 ** n_discordant)
            p = float(min(p, 1.0))
            form = "exact binomial (discordant<25, pure python)"
        else:
            chi2 = (abs(b - c) - 1) ** 2 / n_discordant
            # For df=1, Chi2(x) is equivalent to standard normal Z^2.
            # CDF(x) = erf(sqrt(x/2)). P-value = 1 - CDF = erfc(sqrt(x/2)).
            p = float(math.erfc(math.sqrt(chi2 / 2.0)))
            form = "chi-square with continuity correction (pure python)"

    return {"applicable": True, "b_only_A_correct": b, "c_only_B_correct": c,
            "p_value": p, "form": form,
            "warning": "very few discordant pairs; low power" if n_discordant < 10 else None}



def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Paired non-parametric comparison of per-scenario metric values.
    Appropriate for paired metric arrays (latency per message under config A
    vs B; F1 per scenario A vs B) when normality can't be assumed."""
    if len(a) != len(b):
        return {"applicable": False, "reason": "unequal lengths"}
    diffs = [x - y for x, y in zip(a, b) if x != y]
    if len(diffs) < 6:
        return {"applicable": False,
                "reason": f"only {len(diffs)} non-zero paired differences; "
                          "Wilcoxon needs >=6 for a meaningful p at alpha=0.05"}
    if not _SCIPY:
        return {"applicable": False, "reason": "scipy unavailable"}
    stat, p = _scipy_stats.wilcoxon(a, b)
    return {"applicable": True, "statistic": float(stat), "p_value": float(p), "n_pairs": len(a)}


def paired_t(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Paired t-test. Only appropriate when the paired differences are
    plausibly normal (large n, or metric is an average itself). Includes a
    Shapiro-Wilk normality pre-check and refuses when it clearly fails."""
    if len(a) != len(b) or len(a) < 3:
        return {"applicable": False, "reason": "need paired arrays, n>=3"}
    if not _SCIPY:
        return {"applicable": False, "reason": "scipy unavailable"}
    diffs = [x - y for x, y in zip(a, b)]
    if len(set(diffs)) == 1:
        return {"applicable": False, "reason": "all differences identical"}
    if len(diffs) >= 8:
        w, p_norm = _scipy_stats.shapiro(diffs)
        if p_norm < 0.01:
            return {"applicable": False,
                    "reason": f"differences strongly non-normal (Shapiro p={p_norm:.4f}); "
                              "use wilcoxon_signed_rank instead"}
    stat, p = _scipy_stats.ttest_rel(a, b)
    return {"applicable": True, "statistic": float(stat), "p_value": float(p), "n": len(a)}


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Effect size for paired comparison (Cohen's d on differences)."""
    if len(a) != len(b) or len(a) < 3:
        return {"applicable": False, "reason": "need paired arrays, n>=3"}
    diffs = [x - y for x, y in zip(a, b)]
    mean_d = sum(diffs) / len(diffs)
    var = sum((d - mean_d) ** 2 for d in diffs) / (len(diffs) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return {"applicable": False, "reason": "zero variance in differences"}
    d = mean_d / sd
    mag = ("negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5
           else "medium" if abs(d) < 0.8 else "large")
    return {"applicable": True, "cohens_d": d, "magnitude": mag, "n": len(a)}


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions.
    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    """
    return 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p1)))) - \
           2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))

