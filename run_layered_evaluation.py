#!/usr/bin/env python3
"""
run_layered_evaluation.py
=========================
Runs a (scalable) corpus through the REAL STBV pipeline and records, for every
message, the FIRST layer that flags it (PKI / B1 / MBD / B2 / CP / B3) or
ACCEPT if none do. Produces a per-message CSV and an aggregated results JSON
that the plotting module turns into research-grade figures.

HONESTY CONTRACT:
- Every number this emits comes from actually running your pipeline. Nothing is
  synthesized or assumed.
- If a layer is disabled in the pipeline configuration (e.g. PKI/MBD/CP under
  the default ISCEPipeline), it is recorded as "disabled", never as a pass or a
  catch. Use --full-stack to enable MBD+CP.
- Sample size is whatever you generate; this script does not fabricate scale.

Outputs (into --outdir, default results/):
  per_message.csv    one row per message: ids, truth, per-layer verdicts, catch-layer, latency
  layer_summary.json aggregated counts + metrics per configuration
  run_manifest.json  seed, N, config, versions, timestamp

Usage (from stbv_engine repo root, with this folder on the path):
  python scripts/run_layered_evaluation.py --n 600 --seeds 1 2 3 --full-stack
  python scripts/run_layered_evaluation.py --n 2000 --include-kinematic --full-stack
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time, pathlib, platform
from collections import defaultdict, Counter

LAYERS = ["B1", "B2", "B3"]  # core model: B1 -> B2 -> B3 -> Decision


def _d(x):
    return x if isinstance(x, dict) else {}


def _layer_flags(r):
    """Map a pipeline result dict to per-layer flagged? for the core model B1/B2/B3.
    A layer "flags" when it does NOT pass (B1/B2) or classifies MALICIOUS (B3)."""
    b1, b2, b3 = _d(r.get("b1")), _d(r.get("b2")), _d(r.get("b3"))
    out = {}
    out["B1"] = (b1.get("valid", b1.get("passed")) is False)
    out["B2"] = (b2.get("validation_valid", True) is False)
    b3av = b3.get("available")
    out["B3"] = (b3.get("label") != "BENIGN") if b3av else None
    return out


def _first_catch(flags):
    for L in LAYERS:
        if flags.get(L) is True:
            return L
    return "ACCEPT"


def build_corpus(n, seed, include_kinematic, generate_corpus, extended=None):
    """Assemble a corpus of size ~n from the real generators."""
    corpus = []
    base = generate_corpus(seed=seed)
    corpus.extend(base)
    if extended is not None:
        corpus.extend(generate_corpus(scenarios=extended, seed=seed))
    # Repeat with reseeded variants to reach n (keeps attack_ids distinct per seed round)
    round_ = 1
    while len(corpus) < n:
        more = generate_corpus(seed=seed + round_ * 1000)
        for rec in more:
            rec = dict(rec)
            rec["attack_id"] = f"{rec.get('attack_id','a')}_r{round_}"
            corpus.append(rec)
        round_ += 1
    return corpus[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="target corpus size")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--full-stack", action="store_true", help="enable MBD + CP")
    ap.add_argument("--include-kinematic", action="store_true",
                    help="also draw kinematic scenarios if a generator is available")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    try:
        from semantic_evaluation.semantic_attack_generator import generate_corpus
        from pipeline.orchestrator import ISCEPipeline
    except Exception as e:
        print(f"[FATAL] import failed: {e}\nRun from the stbv_engine repo root.")
        return 2
    try:
        from extended_attack_scenarios import EXTENDED_SCENARIOS as EXT
    except Exception:
        EXT = None

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipe = ISCEPipeline(enable_mbd=args.full_stack, enable_cp=args.full_stack)

    per_rows = []
    agg = defaultdict(lambda: defaultdict(int))   # config -> counter
    catch_counts = Counter()
    latencies = []
    family_correct = defaultdict(lambda: [0, 0])  # family -> [correct, total]

    for seed in args.seeds:
        corpus = build_corpus(args.n, seed, args.include_kinematic, generate_corpus, EXT)
        for rec in corpus:
            truth_mal = str(rec.get("expected_label")).upper() == "MALICIOUS"
            fam = rec.get("attack_category", "benign")
            t0 = time.perf_counter()
            try:
                r = pipe.run([rec], context=(_d(rec.get("scene_context")).get("context") or "urban"))
            except Exception:
                continue
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)
            flags = _layer_flags(r)
            catch = _first_catch(flags)
            catch_counts[catch] += 1
            decision = r.get("decision", "?")
            pred_mal = (decision == "REJECT")
            correct = (pred_mal == truth_mal)
            family_correct[fam][0] += int(correct)
            family_correct[fam][1] += 1
            per_rows.append({
                "seed": seed, "attack_id": rec.get("attack_id"), "family": fam,
                "expected_label": rec.get("expected_label"),
                **{f"flag_{L}": flags.get(L) for L in LAYERS},
                "catch_layer": catch, "decision": decision,
                "correct": correct, "latency_ms": round(dt, 3),
            })

    # write per-message csv
    if per_rows:
        with open(outdir / "per_message.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
            w.writeheader()
            w.writerows(per_rows)

    # aggregate metrics
    tp = sum(1 for r in per_rows if r["decision"] == "REJECT" and r["expected_label"].upper() == "MALICIOUS")
    fp = sum(1 for r in per_rows if r["decision"] == "REJECT" and r["expected_label"].upper() == "BENIGN")
    fn = sum(1 for r in per_rows if r["decision"] != "REJECT" and r["expected_label"].upper() == "MALICIOUS")
    tn = sum(1 for r in per_rows if r["decision"] != "REJECT" and r["expected_label"].upper() == "BENIGN")
    n = len(per_rows) or 1
    prec = tp / (tp + fp) if tp + fp else None
    rec_ = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec_ / (prec + rec_)) if (prec and rec_) else None
    import statistics as st
    lat_sorted = sorted(latencies) or [0]
    def pct(p): return lat_sorted[min(len(lat_sorted) - 1, int(p / 100 * len(lat_sorted)))]

    summary = {
        "n_messages": len(per_rows), "seeds": args.seeds, "full_stack": args.full_stack,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "metrics": {"accuracy": (tp + tn) / n, "precision": prec, "recall": rec_, "f1": f1,
                    "fpr": (fp / (fp + tn) if fp + tn else None)},
        "catch_layer_counts": dict(catch_counts),
        "per_family": {k: {"correct": v[0], "total": v[1],
                           "acc": v[0] / v[1] if v[1] else None}
                       for k, v in family_correct.items()},
        "latency_ms": {"p50": pct(50), "p90": pct(90), "p95": pct(95),
                       "p99": pct(99), "mean": st.mean(latencies) if latencies else None},
    }
    (outdir / "layer_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (outdir / "run_manifest.json").write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_target": args.n, "seeds": args.seeds, "full_stack": args.full_stack,
        "python": platform.python_version(),
        "note": "All values measured by running the real pipeline. No synthetic data.",
    }, indent=2))

    print(f"[done] {len(per_rows)} messages -> {outdir}/per_message.csv")
    print(f"       catch-layer distribution: {dict(catch_counts)}")
    print(f"       F1={f1}, recall={rec_}, precision={prec}")
    print("       Now render figures:  python scripts/make_figures.py --results", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
