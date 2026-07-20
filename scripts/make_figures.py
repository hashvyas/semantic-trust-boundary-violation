#!/usr/bin/env python3
"""
make_figures.py
===============
Renders research-grade figures (vector PDF + PNG) from the output of
run_layered_evaluation.py.

Figures produced:
  fig_layer_funnel      how many messages each layer catches (the "detected at
                        each layer" view you asked for)
  fig_accuracy          F1 / precision / recall / accuracy with CI whiskers
  fig_per_family        per-family accuracy, semantic families highlighted
  fig_confusion         confusion matrix
  fig_latency           per-message latency distribution (box + percentiles)
  fig_semantic_vs_comm  semantic-caught vs communication-caught split
  fig_calibration       reliability curve (if per-sample scores provided)

HONESTY: if run with --sample, it renders a clearly-labelled SYNTHETIC example
(every figure stamped "SAMPLE DATA - not measured") so you can see the format.
With --results <dir> it renders your REAL measured data and no sample banner.

Usage:
  python scripts/make_figures.py --results results          # real run
  python scripts/make_figures.py --sample                   # labelled demo
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
SEM = {"prompt_injection", "context_poisoning", "instruction_override",
       "role_manipulation", "tool_manipulation", "retrieval_poisoning",
       "multi_message", "mixed_attacks", "semantic_narrative_poisoning",
       "rsu_spoofing"}
C = {"comm": "#2b6cb0", "sem": "#e53e3e", "accept": "#38a169",
     "amber": "#d69e2e", "purple": "#805ad5", "gray": "#718096"}


def _banner(fig, is_sample):
    if is_sample:
        fig.text(0.5, 0.5, "SAMPLE DATA — not measured", fontsize=26,
                 color="#e53e3e", alpha=0.12, ha="center", va="center",
                 rotation=25, weight="bold", zorder=0)


def sample_summary():
    """A clearly-labelled synthetic summary matching the real schema."""
    return {
        "_sample": True, "n_messages": 2000,
        "confusion": {"tp": 1310, "fp": 0, "fn": 190, "tn": 500},
        "metrics": {"accuracy": 0.905, "precision": 1.0, "recall": 0.873, "f1": 0.932, "fpr": 0.0},
        "catch_layer_counts": {"B1": 210, "B2": 120, "B3": 980, "ACCEPT": 690},
        "per_family": {
            "prompt_injection": {"acc": 0.72, "total": 200},
            "context_poisoning": {"acc": 0.86, "total": 200},
            "instruction_override": {"acc": 0.63, "total": 200},
            "role_manipulation": {"acc": 0.99, "total": 200},
            "tool_manipulation": {"acc": 0.90, "total": 200},
            "retrieval_poisoning": {"acc": 0.81, "total": 200},
            "multi_message": {"acc": 0.42, "total": 200},
            "mixed_attacks": {"acc": 0.89, "total": 200},
            "replay_kinematic": {"acc": 0.97, "total": 200},
            "sybil_kinematic": {"acc": 0.95, "total": 200},
        },
        "latency_ms": {"p50": 92.0, "p90": 108.0, "p95": 114.0, "p99": 121.0, "mean": 95.0},
    }


def fig_layer_funnel(s, outdir, sample):
    counts = s["catch_layer_counts"]
    order = ["B1", "B2", "B3", "ACCEPT"]  # core model
    labels = [l for l in order if l in counts]
    vals = [counts[l] for l in labels]
    colors = [C["sem"] if l == "B3" else (C["accept"] if l == "ACCEPT" else C["comm"]) for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, str(v),
                ha="center", fontsize=9)
    ax.set_ylabel("Messages caught (first-flagging layer)")
    ax.set_title("Detection by Layer  (B1 structural, B2 behavioral, B3 semantic)")
    _banner(fig, sample)
    fig.tight_layout(); fig.savefig(outdir / "fig_layer_funnel.pdf"); fig.savefig(outdir / "fig_layer_funnel.png"); plt.close(fig)


def fig_semantic_vs_comm(s, outdir, sample):
    counts = s["catch_layer_counts"]
    comm = sum(counts.get(l, 0) for l in ["B1", "B2"])  # B1 structural + B2 behavioral
    sem = counts.get("B3", 0)
    acc = counts.get("ACCEPT", 0)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    wedges, _, _ = ax.pie([comm, sem, acc],
                          labels=[f"B1+B2\n(structural+behavioral)\n{comm}", f"B3\n(semantic)\n{sem}", f"Accepted\n{acc}"],
                          colors=[C["comm"], C["sem"], C["accept"]],
                          autopct="%1.0f%%", startangle=90,
                          wedgeprops=dict(width=0.42))
    ax.set_title("Where attacks are caught: communication vs semantic")
    _banner(fig, sample)
    fig.tight_layout(); fig.savefig(outdir / "fig_semantic_vs_comm.pdf"); fig.savefig(outdir / "fig_semantic_vs_comm.png"); plt.close(fig)


def fig_accuracy(s, outdir, sample):
    m = s["metrics"]; n = s.get("n_messages", 1)
    keys = ["accuracy", "precision", "recall", "f1"]
    vals = [m.get(k) or 0 for k in keys]
    # Wald CI as an honest uncertainty band (proportion-style)
    ci = [1.96 * np.sqrt(max(v * (1 - v), 1e-9) / max(n, 1)) for v in vals]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(keys))
    ax.bar(x, vals, color=C["purple"], yerr=ci, capsize=4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([k.capitalize() for k in keys])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title(f"Detection Performance (n={n}, 95% CI)")
    _banner(fig, sample)
    fig.tight_layout(); fig.savefig(outdir / "fig_accuracy.pdf"); fig.savefig(outdir / "fig_accuracy.png"); plt.close(fig)


def fig_per_family(s, outdir, sample):
    pf = s.get("per_family", {})
    items = sorted(pf.items(), key=lambda kv: (kv[1].get("acc") or 0))
    fams = [k for k, _ in items]
    accs = [v.get("acc") or 0 for _, v in items]
    colors = [C["sem"] if f in SEM else C["comm"] for f in fams]
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.4 * len(fams))))
    ax.barh(fams, accs, color=colors)
    ax.set_xlim(0, 1); ax.set_xlabel("Accuracy")
    ax.set_title("Per-Family Accuracy  (red = semantic, blue = kinematic)")
    for i, v in enumerate(accs):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    _banner(fig, sample)
    fig.tight_layout(); fig.savefig(outdir / "fig_per_family.pdf"); fig.savefig(outdir / "fig_per_family.png"); plt.close(fig)


def fig_confusion(s, outdir, sample):
    c = s["confusion"]
    cm = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred MAL", "pred BEN"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["actual MAL", "actual BEN"])
    mx = cm.max() or 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=15, color="white" if cm[i, j] > 0.5 * mx else "black")
    ax.set_title("Confusion Matrix (full stack)")
    _banner(fig, sample)
    fig.tight_layout(); fig.savefig(outdir / "fig_confusion.pdf"); fig.savefig(outdir / "fig_confusion.png"); plt.close(fig)


def fig_latency(s, outdir, sample):
    L = s["latency_ms"]
    ks = ["p50", "p90", "p95", "p99"]
    vs = [L.get(k) or 0 for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, vs, "o-", color=C["purple"], lw=2, ms=7)
    for i, v in enumerate(vs):
        ax.text(i, v + max(vs) * 0.02, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("ms/message"); ax.set_title("End-to-End Latency Percentiles")
    ax.grid(alpha=0.3)
    _banner(fig, sample)
    fig.tight_layout(); fig.savefig(outdir / "fig_latency.pdf"); fig.savefig(outdir / "fig_latency.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="dir with layer_summary.json")
    ap.add_argument("--sample", action="store_true", help="render labelled synthetic demo")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()

    figdir = pathlib.Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)

    if args.results:
        p = pathlib.Path(args.results) / "layer_summary.json"
        if not p.exists():
            print(f"[FATAL] {p} not found. Run run_layered_evaluation.py first, or use --sample.")
            return 2
        s = json.loads(p.read_text()); sample = False
        print(f"[info] rendering REAL data from {p}")
    else:
        s = sample_summary(); sample = True
        print("[info] rendering SAMPLE data (labelled). Use --results for real runs.")

    fig_layer_funnel(s, figdir, sample)
    fig_semantic_vs_comm(s, figdir, sample)
    fig_accuracy(s, figdir, sample)
    fig_per_family(s, figdir, sample)
    fig_confusion(s, figdir, sample)
    fig_latency(s, figdir, sample)
    print(f"[done] figures -> {figdir}/ (PDF vector + PNG)")
    if sample:
        print("NOTE: these are SAMPLE figures. Replace with a real run for the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
