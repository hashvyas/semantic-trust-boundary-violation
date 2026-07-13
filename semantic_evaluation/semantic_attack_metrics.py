"""
semantic_evaluation/semantic_attack_metrics.py
=================================================
Computes all evaluation metrics from the raw results produced by
semantic_attack_evaluation.py.

Metrics computed
----------------
- Classification: Accuracy, Precision, Recall, F1, FPR, FNR
- Averaging: Macro, Micro, Weighted (per-category)
- Ranking: AUROC, AUPR (using 1 - trust_score as attack score)
- Calibration: ECE (expected calibration error)
- Confidence: B3 confidence distribution (attack vs benign)
- Statistical: McNemar (B1+B2 vs B1+B2+B3), bootstrap CIs, Cohen's h
- Latency: per-stage p50/p95/p99
- Per-category: all of the above, broken down by attack category

Reuses evaluation/stats.py for McNemar and bootstrap CIs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Import existing statistical tests
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.stats import bootstrap_ci, mcnemar

from semantic_evaluation.semantic_attack_dataset import CATEGORY_ORDER


# ----------------------------------------------------------------
# Core confusion matrix
# ----------------------------------------------------------------

def confusion_matrix(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute binary confusion matrix and derived metrics.

    Positive class = attacker message (truth_attacker == True).
    Predicted positive = decision is REJECT.
    """
    tp = sum(1 for r in rows if r["decision"] == "REJECT" and r["truth_attacker"])
    fp = sum(1 for r in rows if r["decision"] == "REJECT" and not r["truth_attacker"])
    fn = sum(1 for r in rows if r["decision"] != "REJECT" and r["truth_attacker"])
    tn = sum(1 for r in rows if r["decision"] != "REJECT" and not r["truth_attacker"])
    n = max(len(rows), 1)
    caution = sum(1 for r in rows if r["decision"] == "CAUTION")
    caution_tp = sum(1 for r in rows if r["decision"] == "CAUTION" and r["truth_attacker"])
    errors = sum(1 for r in rows if r["decision"] == "ERROR")

    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (
        prec == prec and rec == rec and prec + rec > 0) else float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
        "accuracy": (tp + tn) / n,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "fnr": fn / (fn + tp) if (fn + tp) else float("nan"),
        "detection_rate": rec,
        "caution_rate": caution / n,
        "caution_on_attacks": caution_tp,
        "error_rate": errors / n,
    }


# ----------------------------------------------------------------
# Detection including CAUTION (relaxed metric)
# ----------------------------------------------------------------

def detection_with_caution(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Treat both REJECT and CAUTION as 'detected' (relaxed metric).

    This is the 'detection at any level' metric: the system raised
    SOME alarm (not silently ACCEPT).
    """
    detected = sum(1 for r in rows if r["decision"] in ("REJECT", "CAUTION") and r["truth_attacker"])
    total_attacks = sum(1 for r in rows if r["truth_attacker"])
    fp_detected = sum(1 for r in rows if r["decision"] in ("REJECT", "CAUTION") and not r["truth_attacker"])
    total_benign = sum(1 for r in rows if not r["truth_attacker"])
    return {
        "detection_rate_relaxed": detected / total_attacks if total_attacks else float("nan"),
        "false_alarm_rate_relaxed": fp_detected / total_benign if total_benign else float("nan"),
    }


# ----------------------------------------------------------------
# AUROC and AUPR (using 1 - trust_score as attack score)
# ----------------------------------------------------------------

def _auc_trapezoidal(x: List[float], y: List[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return abs(area)


def roc_curve(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute ROC curve and AUROC using (1 - trust_score) as attack score."""
    scored = [(1.0 - r["trust_score"], r["truth_attacker"])
              for r in rows if r["trust_score"] is not None]
    if not scored:
        return {"applicable": False, "reason": "no trust_score values"}

    scored.sort(key=lambda x: x[0])
    n_pos = sum(1 for _, t in scored if t)
    n_neg = len(scored) - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"applicable": False, "reason": "single-class data"}

    fprs, tprs, thresholds = [], [], []
    tp, fp = n_pos, n_neg  # start with everything predicted positive
    fprs.append(fp / n_neg)
    tprs.append(tp / n_pos)
    thresholds.append(scored[0][0] - 0.01)

    for score, truth in scored:
        if truth:
            tp -= 1
        else:
            fp -= 1
        fprs.append(fp / n_neg)
        tprs.append(tp / n_pos)
        thresholds.append(score)

    auroc = _auc_trapezoidal(fprs, tprs)
    return {
        "applicable": True,
        "auroc": auroc,
        "fpr": fprs,
        "tpr": tprs,
        "thresholds": thresholds,
    }


def pr_curve(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Precision-Recall curve and AUPR."""
    scored = [(1.0 - r["trust_score"], r["truth_attacker"])
              for r in rows if r["trust_score"] is not None]
    if not scored:
        return {"applicable": False, "reason": "no trust_score values"}

    scored.sort(key=lambda x: -x[0])  # descending by attack score
    n_pos = sum(1 for _, t in scored if t)
    if n_pos == 0:
        return {"applicable": False, "reason": "no positive samples"}

    precisions, recalls, thresholds = [1.0], [0.0], [scored[0][0] + 0.01]
    tp, fp = 0, 0
    for score, truth in scored:
        if truth:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / n_pos
        precisions.append(prec)
        recalls.append(rec)
        thresholds.append(score)

    aupr = _auc_trapezoidal(recalls, precisions)
    return {
        "applicable": True,
        "aupr": aupr,
        "precision": precisions,
        "recall": recalls,
        "thresholds": thresholds,
    }



# ----------------------------------------------------------------
# Calibration: Expected Calibration Error (ECE)
# ----------------------------------------------------------------

def expected_calibration_error(
    rows: Sequence[Dict[str, Any]], n_bins: int = 10
) -> Dict[str, Any]:
    """Compute ECE for B3's confidence calibration.

    Uses B3's p_malicious as the predicted probability and truth_attacker
    as the ground truth. Only applicable when B3 is available.
    """
    b3_rows = [(r.get("b3_p_malicious"), r["truth_attacker"])
               for r in rows
               if r.get("b3_p_malicious") is not None and r.get("b3_available")]
    if len(b3_rows) < 5:
        return {"applicable": False, "reason": f"only {len(b3_rows)} B3-available rows"}

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bin_data: List[Dict[str, Any]] = []
    ece = 0.0
    for k in range(n_bins):
        lo, hi = bin_edges[k], bin_edges[k + 1]
        in_bin = [(p, t) for p, t in b3_rows if lo <= p < (hi if k < n_bins - 1 else hi + 0.01)]
        if not in_bin:
            bin_data.append({"lo": lo, "hi": hi, "n": 0, "avg_conf": None, "avg_acc": None})
            continue
        avg_conf = sum(p for p, _ in in_bin) / len(in_bin)
        avg_acc = sum(1 for _, t in in_bin if t) / len(in_bin)
        ece += abs(avg_conf - avg_acc) * len(in_bin) / len(b3_rows)
        bin_data.append({"lo": lo, "hi": hi, "n": len(in_bin),
                         "avg_conf": avg_conf, "avg_acc": avg_acc})

    return {"applicable": True, "ece": ece, "bins": bin_data, "n": len(b3_rows)}


# ----------------------------------------------------------------
# B3 Confidence Distribution
# ----------------------------------------------------------------

def confidence_distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract B3 confidence values separated by ground truth."""
    attack_confs = [r["b3_confidence"] for r in rows
                    if r.get("b3_available") and r["truth_attacker"]
                    and r.get("b3_confidence") is not None]
    benign_confs = [r["b3_confidence"] for r in rows
                    if r.get("b3_available") and not r["truth_attacker"]
                    and r.get("b3_confidence") is not None]
    attack_p_mal = [r["b3_p_malicious"] for r in rows
                    if r.get("b3_available") and r["truth_attacker"]
                    and r.get("b3_p_malicious") is not None]
    benign_p_mal = [r["b3_p_malicious"] for r in rows
                    if r.get("b3_available") and not r["truth_attacker"]
                    and r.get("b3_p_malicious") is not None]
    return {
        "attack_confidences": attack_confs,
        "benign_confidences": benign_confs,
        "attack_p_malicious": attack_p_mal,
        "benign_p_malicious": benign_p_mal,
    }


# ----------------------------------------------------------------
# Cohen's h effect size
# ----------------------------------------------------------------

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    """
    return 2.0 * math.asin(math.sqrt(max(0, min(1, p1)))) - \
           2.0 * math.asin(math.sqrt(max(0, min(1, p2))))


# ----------------------------------------------------------------
# Statistical comparison: B1+B2 vs B1+B2+B3
# ----------------------------------------------------------------

def statistical_comparison(
    rows_b1b2: Sequence[Dict[str, Any]],
    rows_full: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """McNemar's test + Cohen's h + bootstrap CIs comparing
    B1+B2 (no B3) vs B1+B2+B3 (full)."""

    # McNemar: paired predictions on the SAME scenarios
    preds_b1b2 = [r["decision"] == "REJECT" for r in rows_b1b2]
    preds_full = [r["decision"] == "REJECT" for r in rows_full]
    truths = [r["truth_attacker"] for r in rows_b1b2]

    mcnemar_result = mcnemar(preds_b1b2, preds_full, truths)

    # Detection rates
    dr_b1b2 = sum(1 for r in rows_b1b2 if r["decision"] == "REJECT" and r["truth_attacker"]) / \
              max(sum(1 for r in rows_b1b2 if r["truth_attacker"]), 1)
    dr_full = sum(1 for r in rows_full if r["decision"] == "REJECT" and r["truth_attacker"]) / \
              max(sum(1 for r in rows_full if r["truth_attacker"]), 1)

    h = cohens_h(dr_full, dr_b1b2)

    # Bootstrap CIs on per-scenario accuracy (1 = correct, 0 = incorrect)
    acc_b1b2 = [1.0 if (r["decision"] == "REJECT") == r["truth_attacker"] else 0.0
                for r in rows_b1b2]
    acc_full = [1.0 if (r["decision"] == "REJECT") == r["truth_attacker"] else 0.0
                for r in rows_full]

    return {
        "mcnemar": mcnemar_result,
        "detection_rate_b1_b2": dr_b1b2,
        "detection_rate_full": dr_full,
        "detection_rate_delta": dr_full - dr_b1b2,
        "cohens_h": h,
        "cohens_h_interpretation": (
            "negligible" if abs(h) < 0.2 else
            "small" if abs(h) < 0.5 else
            "medium" if abs(h) < 0.8 else
            "large"
        ),
        "bootstrap_ci_b1_b2": bootstrap_ci(acc_b1b2),
        "bootstrap_ci_full": bootstrap_ci(acc_full),
    }


# ----------------------------------------------------------------
# Latency summary
# ----------------------------------------------------------------

def latency_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute per-stage latency percentiles."""
    stages: Dict[str, List[float]] = {}
    for r in rows:
        for k, v in (r.get("latencies") or {}).items():
            if isinstance(v, (int, float)):
                stages.setdefault(k, []).append(float(v))

    def pctl(data: List[float], p: float) -> float:
        if not data:
            return float("nan")
        s = sorted(data)
        k = (len(s) - 1) * p
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)

    return {
        stage: {
            "p50": pctl(v, 0.5),
            "p95": pctl(v, 0.95),
            "p99": pctl(v, 0.99),
            "mean": sum(v) / len(v),
            "max": max(v),
        }
        for stage, v in stages.items() if v
    }


# ----------------------------------------------------------------
# Per-category metrics
# ----------------------------------------------------------------

def per_category_metrics(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute confusion matrix per attack category."""
    cats = sorted(set(r.get("attack_category", "") for r in rows))
    result = {}
    for cat in cats:
        cat_rows = [r for r in rows if r.get("attack_category") == cat]
        cm = confusion_matrix(cat_rows)
        det_relaxed = detection_with_caution(cat_rows)
        cm.update(det_relaxed)
        result[cat] = cm
    return result


# ----------------------------------------------------------------
# Compute ALL metrics
# ----------------------------------------------------------------

def compute_all_metrics(
    all_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the complete metrics suite from raw evaluation results.

    Parameters
    ----------
    all_rows : list of dict
        Output from run_evaluation() — rows for ALL configurations.

    Returns
    -------
    dict
        Nested dict of all metrics, organized by configuration.
    """
    configs = sorted(set(r["configuration"] for r in all_rows))
    metrics: Dict[str, Any] = {"configurations": {}}

    for config in configs:
        config_rows = [r for r in all_rows if r["configuration"] == config]
        cm = confusion_matrix(config_rows)
        det_relaxed = detection_with_caution(config_rows)
        roc = roc_curve(config_rows)
        pr = pr_curve(config_rows)
        ece = expected_calibration_error(config_rows)
        conf_dist = confidence_distribution(config_rows)
        latency = latency_summary(config_rows)
        per_cat = per_category_metrics(config_rows)

        metrics["configurations"][config] = {
            "confusion_matrix": cm,
            "detection_with_caution": det_relaxed,
            "roc": {"auroc": roc.get("auroc"), "applicable": roc.get("applicable", False)},
            "pr": {"aupr": pr.get("aupr"), "applicable": pr.get("applicable", False)},
            "calibration": ece,
            "confidence_distribution": {
                "n_attack": len(conf_dist["attack_confidences"]),
                "n_benign": len(conf_dist["benign_confidences"]),
                "mean_attack_p_mal": (sum(conf_dist["attack_p_malicious"]) / len(conf_dist["attack_p_malicious"])
                                      if conf_dist["attack_p_malicious"] else None),
                "mean_benign_p_mal": (sum(conf_dist["benign_p_malicious"]) / len(conf_dist["benign_p_malicious"])
                                      if conf_dist["benign_p_malicious"] else None),
            },
            "latency": latency,
            "per_category": per_cat,
        }

    # Statistical comparison: b1_b2 vs full
    if "b1_b2" in configs and "full" in configs:
        rows_b1b2 = [r for r in all_rows if r["configuration"] == "b1_b2"]
        rows_full = [r for r in all_rows if r["configuration"] == "full"]
        metrics["statistical_comparison"] = statistical_comparison(rows_b1b2, rows_full)

    # ROC/PR curve data (for plotting) — store separately
    metrics["_curve_data"] = {}
    for config in configs:
        config_rows = [r for r in all_rows if r["configuration"] == config]
        metrics["_curve_data"][config] = {
            "roc": roc_curve(config_rows),
            "pr": pr_curve(config_rows),
            "confidence": confidence_distribution(config_rows),
        }

    return metrics
