"""
evaluation/run_experiments.py
================================
Main experiment driver (Parts 5-12 integrated). One command produces the
full results/ logs/ plots/ latex/ tree with manifests, per-message CSVs,
metric tables (CSV + LaTeX), confusion matrices, ROC/PR where scores exist,
ablation plots, latency plots, and paired statistical tests.

Usage:
    python3 evaluation/run_experiments.py                       # default: 3 seeds, all configs+baselines
    python3 evaluation/run_experiments.py --seeds 1 2 3 4 5     # more seeds
    python3 evaluation/run_experiments.py --configs full no_b3  # subset
    python3 evaluation/run_experiments.py --families replay sybil
    python3 evaluation/run_experiments.py --window-cap 10       # bound O(n^2) window growth
    python3 evaluation/run_experiments.py --quick               # tiny smoke run

Notes on honesty/reproducibility:
- Every (seed, configuration) result is traceable via results/manifest.json
  (config, seeds, commit/artifact hash, hardware, versions, dataset SHA-256
  fingerprints).
- If B3's real model is unavailable in the environment, this is recorded in
  the manifest and printed; 'full' and 'no_b3' will then coincide -- the
  driver DETECTS and REPORTS that coincidence rather than presenting the
  two columns as a meaningful comparison.
- window_cap bounds the message window passed to the orchestrator (sliding
  window) purely for tractability; the cap used is recorded in the manifest
  because it affects MBD/CP temporal scope.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.manifest import build_manifest, write_manifest
from evaluation.runner import (BASELINES, CONFIGURATIONS, generate_scenarios_for_seed,
                                 run_scenario)
from evaluation import metrics_and_outputs as mo
from evaluation import stats as st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    ap.add_argument("--configs", nargs="+", default=CONFIGURATIONS + BASELINES)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--message-count", type=int, default=12)
    ap.add_argument("--window-cap", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--quick", action="store_true", help="1 seed, 2 families, short scenarios")
    args = ap.parse_args()

    if args.quick:
        args.seeds = args.seeds[:1]
        args.families = args.families or ["benign", "replay"]
        args.message_count = 6

    out_root = pathlib.Path(args.out)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    results_dir = out_root / run_id
    logs_dir = results_dir / "logs"
    plots_dir = results_dir / "plots"
    latex_dir = results_dir / "latex"
    scen_dir = results_dir / "generated_scenarios"
    for d in (results_dir, logs_dir, plots_dir, latex_dir):
        d.mkdir(parents=True, exist_ok=True)

    # B3 availability -- recorded, never assumed.
    from pipeline.b3_bridge import preload_classifier, classify_text
    preload_classifier()
    b3_available = bool(classify_text("capability probe").get("available"))

    all_rows = []
    t0 = time.perf_counter()
    for seed in args.seeds:
        pairs = generate_scenarios_for_seed(seed, scen_dir, families=args.families,
                                              message_count=args.message_count)
        for cfg_obj, msgs in pairs:
            capped = msgs[: max(args.message_count, 1) * 4]  # generator can emit vehicle_count*message_count
            for configuration in args.configs:
                # Sliding window cap for tractability (recorded in manifest).
                windowed_msgs = capped
                rows = run_scenario(configuration, cfg_obj, windowed_msgs[: args.window_cap * 2])
                all_rows.extend(rows)
                print(f"[seed {seed}] {cfg_obj.scenario_id:28s} {configuration:22s} "
                      f"{len(rows)} msgs  rejects={sum(r['decision']=='REJECT' for r in rows)} "
                      f"errors={sum(r['decision']=='ERROR' for r in rows)}")
    wall = time.perf_counter() - t0

    # ------------------------------------------------------------ outputs --
    mo.write_rows_csv(all_rows, results_dir / "per_message_results.csv")

    # metrics table: configuration x family
    table = {}
    for cfg in args.configs:
        table[cfg] = {}
        fams = sorted({r["family"] for r in all_rows})
        for fam in fams:
            rows = [r for r in all_rows if r["configuration"] == cfg and r["family"] == fam]
            if rows:
                table[cfg][fam] = mo.confusion(rows)
    mo.write_metrics_csv(table, results_dir / "metrics_by_config_family.csv")
    mo.write_latex_table(table, latex_dir / "ablation_table.tex")
    mo.plot_ablation_bars(table, "recall", plots_dir / "ablation_recall.png")
    mo.plot_ablation_bars(table, "fpr", plots_dir / "ablation_fpr.png")
    mo.plot_ablation_bars(table, "caution_rate", plots_dir / "ablation_caution_rate.png")

    # confusion matrices + ROC/PR per configuration (aggregate over families)
    aucs = {}
    for cfg in args.configs:
        rows = [r for r in all_rows if r["configuration"] == cfg]
        if not rows:
            continue
        m = mo.confusion(rows)
        mo.plot_confusion_matrix(m, cfg, plots_dir / f"confusion_{cfg}.png")
        auc = mo.plot_roc_pr(rows, cfg, plots_dir / f"curves_{cfg}")
        if auc:
            aucs[cfg] = auc

    # latency (full config only -- ablations distort stage timing meaningfully)
    full_rows = [r for r in all_rows if r["configuration"] == "full"]
    lat = mo.latency_summary(full_rows)
    (results_dir / "latency_summary.json").write_text(json.dumps(lat, indent=2))
    if lat:
        mo.plot_latency_bars(lat, plots_dir / "latency_p95.png")

    # -------------------------------------------------- statistical tests --
    stats_out = {}
    # Paired McNemar: full vs each ablation, on identical (seed, scenario,
    # msg_index) items.
    def key(r):
        return (r["seed"], r["scenario_id"], r["msg_index"])
    full_by_key = {key(r): r for r in all_rows if r["configuration"] == "full"}
    for cfg in args.configs:
        if cfg == "full":
            continue
        other = {key(r): r for r in all_rows if r["configuration"] == cfg}
        shared = sorted(set(full_by_key) & set(other))
        if not shared:
            continue
        pa = [full_by_key[k]["decision"] in ("REJECT", "CAUTION") for k in shared]
        pb = [other[k]["decision"] in ("REJECT", "CAUTION") for k in shared]
        tt = [full_by_key[k]["truth_attacker"] for k in shared]
        stats_out[f"mcnemar_full_vs_{cfg}"] = st.mcnemar(pa, pb, tt)
    # Bootstrap CI on per-scenario accuracy of the full config (per-scenario
    # is the correct independent sampling unit here; per-run-seed grouping
    # needs >=3 seeds AND is recoverable from per_message_results.csv).
    per_scn_acc = []
    for scn in sorted({r["scenario_id"] for r in all_rows}):
        rows = [r for r in all_rows if r["configuration"] == "full" and r["scenario_id"] == scn]
        if rows:
            per_scn_acc.append(mo.confusion(rows)["accuracy"])
    stats_out["bootstrap_full_accuracy_per_scenario"] = st.bootstrap_ci(per_scn_acc)
    (results_dir / "statistical_tests.json").write_text(json.dumps(stats_out, indent=2))

    # ------------------------------------------------------------ manifest --
    manifest = build_manifest(
        "stbv_full_evaluation",
        config={"seeds": args.seeds, "configurations": args.configs,
                 "families": args.families, "message_count": args.message_count,
                 "window_cap": args.window_cap, "b3_available": b3_available,
                 "wall_seconds": wall,
                 "warning": (None if b3_available else
                              "B3 unavailable: 'full' and 'no_b3' coincide in this run; "
                              "their comparison is NOT meaningful -- rerun on GPU hardware.")},
        seeds=args.seeds,
        dataset_paths=[scen_dir],
    )
    write_manifest(results_dir, manifest)

    print(f"\nDone in {wall:.1f}s. Results tree: {results_dir}")
    if not b3_available:
        print("WARNING: B3 real model unavailable -- see manifest warning. "
              "'full' vs 'no_b3' comparison is not meaningful in this run.")
    n_errors = sum(1 for r in all_rows if r["decision"] == "ERROR")
    print(f"Rows: {len(all_rows)}  |  ERROR rows (explicit, not silent): {n_errors}")
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
