"""
semantic_evaluation/semantic_attack_plots.py
===============================================
Publication-quality figures for the semantic attack evaluation.

All plots are saved as both PNG (300 DPI) and PDF (vector) in the
output directory.

Figures generated
-----------------
1. Confusion matrix heatmaps (3 panels)
2. ROC curve overlay (B1+B2 vs B1+B2+B3)
3. PR curve overlay
4. Per-category detection rate bar chart
5. B3 confidence distribution histogram
6. Calibration reliability diagram
7. Ablation waterfall
8. Latency distribution box plots
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from semantic_evaluation.semantic_attack_dataset import CATEGORY_ORDER


# ----------------------------------------------------------------
# Style setup
# ----------------------------------------------------------------

def _setup_style() -> None:
    """Configure publication-quality matplotlib style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _save(fig: plt.Figure, out_dir: pathlib.Path, name: str) -> None:
    """Save figure as PNG and PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------
# 1. Confusion Matrix Heatmaps
# ----------------------------------------------------------------

def plot_confusion_matrices(
    metrics: Dict[str, Any], out_dir: pathlib.Path
) -> None:
    """3-panel confusion matrix heatmaps."""
    _setup_style()
    configs = ["b1_only", "b1_b2", "full"]
    labels_display = ["B1 Only", "B1+B2", "B1+B2+B3"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, config, label in zip(axes, configs, labels_display):
        cm_data = metrics["configurations"].get(config, {}).get("confusion_matrix", {})
        tp = cm_data.get("tp", 0)
        fp = cm_data.get("fp", 0)
        fn = cm_data.get("fn", 0)
        tn = cm_data.get("tn", 0)

        matrix = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Attack"])
        ax.set_yticklabels(["Benign", "Attack"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{label}\n(Acc: {cm_data.get('accuracy', 0):.1%}, "
                     f"F1: {cm_data.get('f1', 0):.3f})")

        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                       fontsize=14, fontweight="bold", color=color)

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Confusion Matrices — Semantic Attack Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "confusion_matrix")


# ----------------------------------------------------------------
# 2. ROC Curve Overlay
# ----------------------------------------------------------------

def plot_roc_curves(
    metrics: Dict[str, Any], out_dir: pathlib.Path
) -> None:
    """ROC curves for B1+B2 and B1+B2+B3."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {"b1_b2": "#2196F3", "full": "#E91E63"}
    labels = {"b1_b2": "B1+B2 (no semantic)", "full": "B1+B2+B3 (full)"}

    for config in ["b1_b2", "full"]:
        curve_data = metrics.get("_curve_data", {}).get(config, {}).get("roc", {})
        if not curve_data.get("applicable"):
            continue
        fprs = curve_data["fpr"]
        tprs = curve_data["tpr"]
        auroc = curve_data["auroc"]
        ax.plot(fprs, tprs, color=colors.get(config, "gray"),
                label=f"{labels.get(config, config)} (AUROC={auroc:.3f})",
                linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Semantic Attack Detection", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    _save(fig, out_dir, "roc_curve")


# ----------------------------------------------------------------
# 3. PR Curve Overlay
# ----------------------------------------------------------------

def plot_pr_curves(
    metrics: Dict[str, Any], out_dir: pathlib.Path
) -> None:
    """Precision-Recall curves."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {"b1_b2": "#2196F3", "full": "#E91E63"}
    labels = {"b1_b2": "B1+B2 (no semantic)", "full": "B1+B2+B3 (full)"}

    for config in ["b1_b2", "full"]:
        curve_data = metrics.get("_curve_data", {}).get(config, {}).get("pr", {})
        if not curve_data.get("applicable"):
            continue
        recalls = curve_data["recall"]
        precisions = curve_data["precision"]
        aupr = curve_data["aupr"]
        ax.plot(recalls, precisions, color=colors.get(config, "gray"),
                label=f"{labels.get(config, config)} (AUPR={aupr:.3f})",
                linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Semantic Attack Detection", fontweight="bold")
    ax.legend(loc="lower left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    _save(fig, out_dir, "pr_curve")


# ----------------------------------------------------------------
# 4. Per-Category Detection Rate Bar Chart
# ----------------------------------------------------------------

def plot_detection_by_category(
    metrics: Dict[str, Any], out_dir: pathlib.Path
) -> None:
    """Grouped bar chart: detection rate per category per configuration."""
    _setup_style()

    configs = ["b1_only", "b1_b2", "full"]
    config_labels = {"b1_only": "B1 Only", "b1_b2": "B1+B2", "full": "B1+B2+B3"}
    colors = {"b1_only": "#90CAF9", "b1_b2": "#2196F3", "full": "#E91E63"}

    # Only attack categories (exclude benign_controls for detection rate)
    cats = [c for c in CATEGORY_ORDER if c != "benign_controls"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(cats))
    width = 0.25

    for i, config in enumerate(configs):
        per_cat = metrics["configurations"].get(config, {}).get("per_category", {})
        rates = []
        for cat in cats:
            cat_data = per_cat.get(cat, {})
            rates.append(cat_data.get("detection_rate", 0.0))
        # Handle NaN
        rates = [r if r == r else 0.0 for r in rates]
        bars = ax.bar(x + i * width, rates, width,
                      label=config_labels.get(config, config),
                      color=colors.get(config, "gray"),
                      edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f"{rate:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Attack Category")
    ax.set_ylabel("Detection Rate (Recall)")
    ax.set_title("Detection Rate by Attack Category — B3 Contribution", fontweight="bold")
    ax.set_xticks(x + width)
    cat_labels = [c.replace("_", "\n") for c in cats]
    ax.set_xticklabels(cat_labels, fontsize=8)
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    _save(fig, out_dir, "detection_by_category")


# ----------------------------------------------------------------
# 5. B3 Confidence Distribution
# ----------------------------------------------------------------

def plot_confidence_distribution(
    metrics: Dict[str, Any], out_dir: pathlib.Path
) -> None:
    """Histogram of B3's p(MALICIOUS) for attacks vs benign."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    full_conf = metrics.get("_curve_data", {}).get("full", {}).get("confidence", {})
    attack_p = full_conf.get("attack_p_malicious", [])
    benign_p = full_conf.get("benign_p_malicious", [])

    bins = np.linspace(0, 1, 21)
    if attack_p:
        ax.hist(attack_p, bins=bins, alpha=0.7, color="#E91E63",
                label=f"Attack (n={len(attack_p)})", edgecolor="white")
    if benign_p:
        ax.hist(benign_p, bins=bins, alpha=0.7, color="#4CAF50",
                label=f"Benign (n={len(benign_p)})", edgecolor="white")

    # Add threshold lines
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Decision boundary (0.5)")
    ax.axvline(x=0.85, color="red", linestyle=":", alpha=0.5, label="High-confidence threshold (0.85)")

    ax.set_xlabel("B3 P(MALICIOUS)")
    ax.set_ylabel("Count")
    ax.set_title("B3 Confidence Distribution — Attack vs Benign", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "confidence_distribution")


# ----------------------------------------------------------------
# 6. Calibration Reliability Diagram
# ----------------------------------------------------------------

def plot_calibration(
    metrics: Dict[str, Any], out_dir: pathlib.Path
) -> None:
    """Reliability diagram for B3's calibration."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 6))

    cal = metrics["configurations"].get("full", {}).get("calibration", {})
    if not cal.get("applicable"):
        ax.text(0.5, 0.5, "Calibration data not available",
               transform=ax.transAxes, ha="center")
        _save(fig, out_dir, "calibration_reliability")
        return

    bins = cal.get("bins", [])
    confs = [b["avg_conf"] for b in bins if b.get("avg_conf") is not None]
    accs = [b["avg_acc"] for b in bins if b.get("avg_acc") is not None]
    counts = [b["n"] for b in bins if b.get("avg_conf") is not None]

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.bar(confs, accs, width=0.08, alpha=0.7, color="#2196F3",
           edgecolor="white", label="B3 calibration")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration — ECE = {cal.get('ece', 0):.4f}", fontweight="bold")
    ax.legend()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    _save(fig, out_dir, "calibration_reliability")


# ----------------------------------------------------------------
# 7. Ablation Waterfall
# ----------------------------------------------------------------

def plot_ablation_waterfall(
    metrics: Dict[str, Any], out_dir: pathlib.Path
) -> None:
    """Waterfall chart showing incremental detection rate gain per layer."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = ["b1_only", "b1_b2", "full"]
    labels = ["B1 Only", "+B2\n(B1+B2)", "+B3\n(B1+B2+B3)"]
    colors_bar = ["#90CAF9", "#2196F3", "#E91E63"]

    rates = []
    for config in configs:
        cm = metrics["configurations"].get(config, {}).get("confusion_matrix", {})
        dr = cm.get("detection_rate", 0.0)
        rates.append(dr if dr == dr else 0.0)

    # Compute increments
    increments = [rates[0]]
    for i in range(1, len(rates)):
        increments.append(rates[i] - rates[i - 1])

    bottoms = [0.0]
    for i in range(1, len(rates)):
        bottoms.append(rates[i - 1])

    x = range(len(labels))
    for i, (xi, inc, bot) in enumerate(zip(x, increments, bottoms)):
        color = colors_bar[i]
        ax.bar(xi, inc, bottom=bot, color=color, edgecolor="white",
               width=0.6, label=labels[i])
        # Label
        ax.text(xi, bot + inc / 2, f"+{inc:.1%}" if i > 0 else f"{inc:.1%}",
               ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        # Total at top
        ax.text(xi, bot + inc + 0.02, f"{rates[i]:.1%}",
               ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Detection Rate (Recall)")
    ax.set_title("Ablation — Incremental Detection Rate Contribution", fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, max(rates) * 1.2 + 0.1)
    plt.tight_layout()
    _save(fig, out_dir, "ablation_waterfall")


# ----------------------------------------------------------------
# 8. Latency Distribution
# ----------------------------------------------------------------

def plot_latency_distribution(
    metrics: Dict[str, Any], out_dir: pathlib.Path
) -> None:
    """Box plot of per-stage latencies for the full pipeline."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    latency = metrics["configurations"].get("full", {}).get("latency", {})
    if not latency:
        ax.text(0.5, 0.5, "No latency data", transform=ax.transAxes, ha="center")
        _save(fig, out_dir, "latency_distribution")
        return

    stages = sorted(latency.keys())
    data_for_box = []
    stage_labels = []
    for stage in stages:
        stats = latency[stage]
        # Reconstruct approximate data from percentiles for box plot
        data_for_box.append([stats["p50"], stats["p95"], stats["p99"],
                            stats["mean"], stats["max"]])
        stage_labels.append(stage.replace("_ms", "").replace("_", "\n"))

    # Use bar chart of p50 with error bars to p95
    p50s = [latency[s]["p50"] for s in stages]
    p95s = [latency[s]["p95"] for s in stages]
    errs = [p95 - p50 for p50, p95 in zip(p50s, p95s)]

    x = range(len(stages))
    ax.bar(x, p50s, color="#2196F3", alpha=0.8, label="p50 (median)")
    ax.errorbar(x, p50s, yerr=[([0] * len(errs)), errs], fmt="none",
                capsize=5, color="black", label="p95")

    ax.set_xticks(list(x))
    ax.set_xticklabels(stage_labels, fontsize=8)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Stage Latency — Full Pipeline (B1+B2+B3)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "latency_distribution")


# ----------------------------------------------------------------
# Generate ALL plots
# ----------------------------------------------------------------

def generate_all_plots(
    metrics: Dict[str, Any], out_dir: pathlib.Path
) -> List[str]:
    """Generate all publication figures. Returns list of filenames created."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []

    plot_confusion_matrices(metrics, out_dir)
    files.append("confusion_matrix")

    plot_roc_curves(metrics, out_dir)
    files.append("roc_curve")

    plot_pr_curves(metrics, out_dir)
    files.append("pr_curve")

    plot_detection_by_category(metrics, out_dir)
    files.append("detection_by_category")

    plot_confidence_distribution(metrics, out_dir)
    files.append("confidence_distribution")

    plot_calibration(metrics, out_dir)
    files.append("calibration_reliability")

    plot_ablation_waterfall(metrics, out_dir)
    files.append("ablation_waterfall")

    plot_latency_distribution(metrics, out_dir)
    files.append("latency_distribution")

    return files
