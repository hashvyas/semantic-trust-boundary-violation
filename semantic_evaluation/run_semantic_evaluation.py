"""
semantic_evaluation/run_semantic_evaluation.py
================================================
Single-command runner for the semantic attack evaluation framework.

Executes:
1. Generates/loads the semantic V2X message corpus.
2. Evaluates the corpus against the three pipeline configurations
   (B1-only, B1+B2, B1+B2+B3).
3. Computes the complete metrics suite (including statistical tests).
4. Generates publication-quality figures (PNG/PDF).
5. Exports results to JSON, CSV, and LaTeX tables.
6. Prints a clean, comprehensive summary table of the findings.

Usage:
  python semantic_evaluation/run_semantic_evaluation.py [--out results/semantic] [--quick]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.b3_bridge import preload_classifier
from semantic_evaluation.semantic_attack_dataset import (
    ALL_SCENARIOS,
    CATEGORY_ORDER,
)
from semantic_evaluation.semantic_attack_evaluation import run_evaluation
from semantic_evaluation.semantic_attack_metrics import compute_all_metrics
from semantic_evaluation.semantic_attack_plots import generate_all_plots


def export_raw_results_csv(rows: List[Dict[str, Any]], path: pathlib.Path) -> None:
    """Save raw scenario results to a flat CSV."""
    if not rows:
        return
    # Omit columns with large nested data to keep CSV clean
    exclude_keys = {"latencies", "synthesized_text", "reasoning"}
    headers = [k for k in rows[0].keys() if k not in exclude_keys]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            row_to_write = {k: v for k, v in r.items() if k in headers}
            writer.writerow(row_to_write)


def export_latex_tables(metrics: Dict[str, Any], out_dir: pathlib.Path) -> None:
    """Generate LaTeX tables suitable for an IEEE paper."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main results comparison table
    main_tex_path = out_dir / "main_results_table.tex"
    with main_tex_path.open("w", encoding="utf-8") as fh:
        fh.write(r"""\begin{table}[h]
\centering
\caption{Overall Performance Comparison on Semantic Attacks}
\label{tab:semantic_results}
\begin{tabular}{lcccccc}
\hline
\textbf{Configuration} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{FPR} & \textbf{Caution Rate} \\ \hline
""")
        configs_order = ["b1_only", "b1_b2", "full"]
        config_names = {"b1_only": "B1 Only (Structural)", "b1_b2": "B1 + B2 (Explainability)", "full": "B1 + B2 + B3 (Semantic)"}

        for cfg in configs_order:
            cfg_data = metrics["configurations"].get(cfg, {})
            cm = cfg_data.get("confusion_matrix", {})
            name = config_names.get(cfg, cfg)
            acc = cm.get("accuracy", 0.0)
            prec = cm.get("precision", 0.0)
            rec = cm.get("recall", 0.0)
            f1 = cm.get("f1", 0.0)
            fpr = cm.get("fpr", 0.0)
            caution = cm.get("caution_rate", 0.0)

            # Handle NaNs in precision/F1 if no positive predictions
            prec_str = f"{prec:.3f}" if prec == prec else "N/A"
            f1_str = f"{f1:.3f}" if f1 == f1 else "N/A"

            fh.write(f"{name} & {acc:.3f} & {prec_str} & {rec:.3f} & {f1_str} & {fpr:.3f} & {caution:.3f} \\\\\n")

        fh.write(r"""\hline
\end{tabular}
\end{table}
""")

    # 2. Per-category detection rate comparison table
    cat_tex_path = out_dir / "per_category_table.tex"
    with cat_tex_path.open("w", encoding="utf-8") as fh:
        fh.write(r"""\begin{table}[h]
\centering
\caption{Detection Rates (Recall) by Semantic Attack Category}
\label{tab:per_category_semantic}
\begin{tabular}{lccc}
\hline
\textbf{Attack Category} & \textbf{B1 Only (Recall)} & \textbf{B1 + B2 (Recall)} & \textbf{B1 + B2 + B3 (Recall)} \\ \hline
""")
        for cat in CATEGORY_ORDER:
            if cat == "benign_controls":
                continue
            recalls = {}
            for cfg in ["b1_only", "b1_b2", "full"]:
                cat_data = metrics["configurations"].get(cfg, {}).get("per_category", {}).get(cat, {})
                rec = cat_data.get("recall", 0.0)
                recalls[cfg] = f"{rec:.1%}" if rec == rec else "0.0%"

            cat_display = cat.replace("_", " ").title()
            fh.write(f"{cat_display} & {recalls['b1_only']} & {recalls['b1_b2']} & {recalls['full']} \\\\\n")

        fh.write(r"""\hline
\end{tabular}
\end{table}
""")

    # 3. Statistical tests table
    stats_tex_path = out_dir / "statistical_tests_table.tex"
    if "statistical_comparison" in metrics:
        sc = metrics["statistical_comparison"]
        mcn = sc.get("mcnemar", {})
        boot_b1b2 = sc.get("bootstrap_ci_b1_b2", {})
        boot_full = sc.get("bootstrap_ci_full", {})

        with stats_tex_path.open("w", encoding="utf-8") as fh:
            fh.write(r"""\begin{table}[h]
\centering
\caption{Statistical Significance and Effect Size Analysis}
\label{tab:semantic_stats}
\begin{tabular}{lc}
\hline
\textbf{Statistical Metric} & \textbf{Value / Outcome} \\ \hline
""")
            fh.write(f"McNemar's p-value (B1+B2 vs. Full) & {mcn.get('p_value', 'N/A')} ({mcn.get('form', 'N/A')}) \\\\\n")
            fh.write(f"Detection Rate Delta & {sc.get('detection_rate_delta', 0.0):+.1%} \\\\\n")
            fh.write(f"Cohen's $h$ Effect Size (Detection Rate) & {sc.get('cohens_h', 0.0):.3f} ({sc.get('cohens_h_interpretation', 'N/A')}) \\\\\n")
            fh.write(f"B1+B2 Bootstrap Accuracy CI & [{boot_b1b2.get('ci_low', 0.0):.3f}, {boot_b1b2.get('ci_high', 0.0):.3f}] \\\\\n")
            fh.write(f"Full Pipeline Bootstrap Accuracy CI & [{boot_full.get('ci_low', 0.0):.3f}, {boot_full.get('ci_high', 0.0):.3f}] \\\\\n")
            fh.write(r"""\hline
\end{tabular}
\end{table}
""")


def print_summary_report(metrics: Dict[str, Any]) -> None:
    """Print a clean ASCII summary table of the evaluation."""
    print("\n" + "="*80)
    print("                    SEMANTIC EVALUATION PERFORMANCE REPORT")
    print("="*80)

    configs_order = ["b1_only", "b1_b2", "full"]
    config_labels = {
        "b1_only": "B1 Only (Structural Check)",
        "b1_b2": "B1+B2 (Explainability Layer)",
        "full": "B1+B2+B3 (Full Semantic Gate)",
    }

    # Print general stats
    print(f"{'Pipeline Configuration':30s} | {'Acc':6s} | {'Prec':6s} | {'Recall':6s} | {'F1':6s} | {'FPR':6s} | {'Caution':7s}")
    print("-" * 80)
    for cfg in configs_order:
        cfg_data = metrics["configurations"].get(cfg, {})
        cm = cfg_data.get("confusion_matrix", {})
        acc = cm.get("accuracy", 0.0)
        prec = cm.get("precision", 0.0)
        rec = cm.get("recall", 0.0)
        f1 = cm.get("f1", 0.0)
        fpr = cm.get("fpr", 0.0)
        caution = cm.get("caution_rate", 0.0)

        prec_str = f"{prec:.3f}" if prec == prec else "N/A"
        f1_str = f"{f1:.3f}" if f1 == f1 else "N/A"

        label = config_labels.get(cfg, cfg)
        print(f"{label:30s} | {acc:.3f}  | {prec_str:6s} | {rec:.3f}  | {f1_str:6s} | {fpr:.3f}  | {caution:.3f}")
    print("-" * 80)

    # Print per-category recall
    print("\nRecall (Detection Rate) by Attack Category:")
    print("-" * 80)
    print(f"{'Attack Category':30s} | {'B1 Only':10s} | {'B1+B2':10s} | {'B1+B2+B3 (Full)':15s}")
    print("-" * 80)

    for cat in CATEGORY_ORDER:
        if cat == "benign_controls":
            continue
        recalls = {}
        for cfg in ["b1_only", "b1_b2", "full"]:
            cat_data = metrics["configurations"].get(cfg, {}).get("per_category", {}).get(cat, {})
            rec = cat_data.get("recall", 0.0)
            recalls[cfg] = f"{rec:.1%}" if rec == rec else "0.0%"
        cat_display = cat.replace("_", " ").title()
        print(f"{cat_display:30s} | {recalls['b1_only']:10s} | {recalls['b1_b2']:10s} | {recalls['full']:15s}")
    print("-" * 80)

    # Print statistical comparison if available
    if "statistical_comparison" in metrics:
        sc = metrics["statistical_comparison"]
        mcn = sc.get("mcnemar", {})
        print("\nStatistical Analysis (B1+B2 vs. B1+B2+B3):")
        print(f"  McNemar Test p-value  : {mcn.get('p_value', 'N/A')} ({mcn.get('form', 'N/A')})")
        print(f"  Detection Rate Delta  : {sc.get('detection_rate_delta', 0.0):+.1%}")
        print(f"  Cohen's h Effect Size : {sc.get('cohens_h', 0.0):.3f} ({sc.get('cohens_h_interpretation', 'N/A')})")
    print("="*80 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Semantic Attack Evaluation Framework")
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "semantic"),
        help="Directory to save evaluation reports and plots",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Execute quick smoke-test (subset of scenarios)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Deterministic scenario generation seed",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    # Append timestamped folder
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Semantic Attack Evaluation...")
    print(f"Output directory: {run_dir.resolve()}")

    # Preload B3 model
    print("Preloading B3 classifier...")
    preload_classifier()

    # Filter scenarios if in --quick mode to smoke-test all paths quickly
    scenarios_to_evaluate = ALL_SCENARIOS
    if args.quick:
        print("Quick smoke-test mode enabled. Evaluating a subset of scenarios...")
        # Take 1 scenario from each category
        quick_scenarios = []
        for cat in CATEGORY_ORDER:
            cat_scens = [s for s in ALL_SCENARIOS if s.category == cat]
            if cat_scens:
                quick_scenarios.append(cat_scens[0])
        scenarios_to_evaluate = quick_scenarios
        print(f"Selected {len(scenarios_to_evaluate)} scenarios for quick evaluation.")

    # 1. Run evaluation
    t_start = time.perf_counter()
    raw_rows = run_evaluation(
        scenarios=scenarios_to_evaluate,
        seed=args.seed,
        verbose=True,
    )
    t_eval = time.perf_counter() - t_start
    print(f"Pipeline execution complete in {t_eval:.2f} seconds.")

    # 2. Compute metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(raw_rows)

    # 3. Save raw results
    print("Saving raw results and metadata...")
    with (run_dir / "raw_results.json").open("w", encoding="utf-8") as fh:
        json.dump(raw_rows, fh, indent=2, default=str)

    export_raw_results_csv(raw_rows, run_dir / "raw_results.csv")

    # Serialize metrics (excluding large plotting coordinates/arrays)
    metrics_to_save = {k: v for k, v in metrics.items() if k != "_curve_data"}
    with (run_dir / "metrics_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_to_save, fh, indent=2, default=str)

    # Save per-category summary as CSV
    per_cat_csv_path = run_dir / "metrics_per_category.csv"
    with per_cat_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Configuration", "Category", "n", "TP", "FP", "FN", "TN", "Recall", "Precision", "F1", "FPR"])
        for config, cfg_data in metrics["configurations"].items():
            per_cat = cfg_data.get("per_category", {})
            for cat, cm in per_cat.items():
                writer.writerow([
                    config, cat, cm.get("n"), cm.get("tp"), cm.get("fp"), cm.get("fn"), cm.get("tn"),
                    f"{cm.get('recall', 0.0):.4f}", f"{cm.get('precision', 0.0):.4f}", f"{cm.get('f1', 0.0):.4f}", f"{cm.get('fpr', 0.0):.4f}"
                ])

    # 4. Generate plots
    print("Generating publication figures (PNG and PDF)...")
    created_plots = generate_all_plots(metrics, run_dir / "plots")
    print(f"Generated plots: {', '.join(created_plots)}")

    # 5. Export LaTeX tables
    print("Generating LaTeX tables...")
    export_latex_tables(metrics, run_dir / "latex")

    # 6. Print final report to console
    print_summary_report(metrics)

    # Create a link file pointing to the latest run directory
    latest_link = out_dir / "latest"
    if latest_link.exists():
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.is_dir():
            # On Windows, sometimes symlinks aren't configured or we just write a text file
            try:
                latest_link.unlink()
            except Exception:
                import shutil
                shutil.rmtree(latest_link)
        else:
            latest_link.unlink()

    try:
        # Write direct path to a file for easy reading
        (out_dir / "latest_run_path.txt").write_text(str(run_dir.resolve()), encoding="utf-8")
    except Exception:
        pass

    print(f"All outputs saved to: {run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
