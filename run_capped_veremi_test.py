#!/usr/bin/env python3
"""
run_capped_veremi_test.py
==========================
Runs a capped evaluation of the V2X Security/Trust Pipeline on processed VeReMi data
and outputs detailed results and metrics directly to the terminal.
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

# Ensure workspace root is in path
ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.orchestrator import ISCEPipeline
from b1_scsv.scsv import SCSV


def list_available_datasets(base_dir: pathlib.Path) -> List[str]:
    if not base_dir.is_dir():
        return []
    datasets = []
    for d in base_dir.iterdir():
        if d.is_dir() and (d / "veremi_flat_reports.json").is_file():
            datasets.append(d.name)
    return sorted(datasets)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V2X trust pipeline on a capped subset of VeReMi data.")
    parser.add_argument("--dataset", help="Name of the preprocessed dataset subdirectory in data/veremi_processed")
    parser.add_argument("--cap", type=int, default=30, help="Number of messages to run (default: 30)")
    parser.add_argument("--window-size", type=float, default=5.0, help="Temporal window size in seconds for sliding window (default: 5.0)")
    parser.add_argument("--context", default="urban", choices=["urban", "rural", "highway"], help="Road context to pass to pipeline (default: urban)")
    args = parser.parse_args()

    processed_root = ROOT / "data" / "veremi_processed"
    available = list_available_datasets(processed_root)

    if not available:
        print(f"[FATAL] No preprocessed datasets found under {processed_root}")
        print("        Please run import_veremi.py first to import raw logs.")
        return 1

    dataset_name = args.dataset
    if not dataset_name:
        # Default to a quick or tiny dataset if available
        quick_candidates = [d for d in available if "quick" in d or "tiny" in d]
        dataset_name = quick_candidates[0] if quick_candidates else available[0]

    if dataset_name not in available:
        print(f"[FATAL] Dataset '{dataset_name}' not found.")
        print(f"        Available datasets: {', '.join(available)}")
        return 2

    dataset_dir = processed_root / dataset_name
    reports_file = dataset_dir / "veremi_flat_reports.json"
    manifest_file = dataset_dir / "manifest.json"

    print("=" * 80)
    print(f"RUNNING V2X TRUST PIPELINE CAPPED TEST ON VEREMI DATA")
    print("=" * 80)
    print(f"Dataset:      {dataset_name}")
    print(f"Input file:   {reports_file}")
    if manifest_file.is_file():
        try:
            manifest = json.loads(manifest_file.read_text())
            print(f"Label rule:   {manifest.get('label_rule')}")
            print(f"Schema:       {manifest.get('schema')}")
        except Exception:
            pass
    print(f"Road Context: {args.context}")
    print(f"Message Cap:  {args.cap}")
    print(f"Window Size:  {args.window_size} seconds")
    print("-" * 80)

    # Load messages
    print("[1/3] Loading flat reports...")
    try:
        with open(reports_file, "r") as f:
            all_reports = json.load(f)
    except Exception as e:
        print(f"[FATAL] Failed to read reports file: {e}")
        return 3

    total_available = len(all_reports)
    print(f"      Loaded {total_available} messages total.")
    run_count = min(args.cap, total_available)
    reports_to_run = all_reports[:run_count]
    print(f"      Will evaluate the first {run_count} messages.")

    # Initialize Pipeline
    print("[2/3] Initializing Trust Pipeline...")
    t_init_start = time.perf_counter()
    pipeline = ISCEPipeline(
        scsv=SCSV(cert_rotation_owner="mbd"),
        enable_mbd=True,
        enable_cp=True,
    )
    init_duration = time.perf_counter() - t_init_start
    print(f"      Pipeline initialized in {init_duration:.3f}s")
    b3_status = "REAL" if pipeline.b3_load_ms > 0 else "STUB (Unavailable, fallback active)"
    print(f"      B3 Semantic Gate Classifier: {b3_status}")
    print("-" * 80)

    # Process messages statefully/sliding window
    print("[3/3] Processing message stream...")
    results = []
    
    # Store processed messages for sliding window
    history: List[Dict[str, Any]] = []

    # Table header
    print(f"{'Idx':<4} | {'Sender':<6} | {'Time (s)':<10} | {'Truth':<7} | {'B1 Valid':<8} | {'MBD Pass':<8} | {'B3 Risk':<8} | {'Decision':<8} | {'Latency':<8}")
    print("-" * 80)

    for idx, target_msg in enumerate(reports_to_run):
        history.append(target_msg)
        
        # Build sliding window based on timestamp range
        target_time = target_msg["timestamp"]
        window = [
            m for m in history
            if abs(target_time - m["timestamp"]) <= args.window_size
        ]
        
        # Run the pipeline
        try:
            r = pipeline.run(window, context=args.context)
            latency_ms = r["latencies"]["total_ms"]
            
            # Extract info
            sender = target_msg["sender"]
            timestamp = target_msg["timestamp"]
            truth = "ATTACK" if target_msg.get("is_attacker", False) else "GENUINE"
            
            b1_valid = "PASS" if r["b1"]["valid"] else "FAIL"
            mbd_pass = "PASS" if (r["mbd"]["passed"] if r["mbd"] else True) else "FAIL"
            b3_risk = r["b3"].get("risk_level", "n/a")
            decision = r["decision"]
            
            # Print row
            print(f"{idx:<4} | {sender:<6} | {timestamp:<10.3f} | {truth:<7} | {b1_valid:<8} | {mbd_pass:<8} | {b3_risk:<8} | {decision:<8} | {latency_ms:<6.1f}ms")
            
            results.append({
                "index": idx,
                "sender": sender,
                "timestamp": timestamp,
                "truth_attacker": target_msg.get("is_attacker", False),
                "decision": decision,
                "latency_ms": latency_ms,
                "pipeline_result": r
            })
        except Exception as e:
            print(f"{idx:<4} | ERROR processing message: {e}")
            import traceback
            traceback.print_exc()

    print("-" * 80)
    print("EVALUATION METRICS & RESULTS SUMMARY")
    print("-" * 80)

    # Compute metrics
    total = len(results)
    if total == 0:
        print("No messages processed successfully.")
        return 0

    tp = fp = tn = fn = 0
    total_lat = 0.0

    for res in results:
        is_atk = res["truth_attacker"]
        dec = res["decision"]
        total_lat += res["latency_ms"]

        predicted_attack = dec in ("REJECT", "CAUTION")

        if is_atk:
            if predicted_attack:
                tp += 1
            else:
                fn += 1
        else:
            if predicted_attack:
                fp += 1
            else:
                tn += 1

    avg_lat = total_lat / total
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Total evaluated:      {total}")
    print(f"Genuine messages:     {tn + fp} (Correctly accepted: {tn}, False alarm/cautioned or rejected: {fp})")
    print(f"Attacker messages:    {tp + fn} (Correctly cautioned/rejected: {tp}, Missed/accepted: {fn})")
    print(f"Confusion Matrix:")
    print(f"                      Predicted Benign (ACCEPT)    Predicted Attack (CAUTION/REJECT)")
    print(f"  Actual GENUINE      {tn:<27}  {fp:<15}")
    print(f"  Actual ATTACKER     {fn:<27}  {tp:<15}")
    print("-" * 80)
    print(f"Accuracy:             {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:            {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall (Sensitivity):  {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:             {f1:.4f}")
    print(f"Average Latency:      {avg_lat:.2f} ms")
    print("-" * 80)
    print("Evaluation Mapping:")
    print("    ACCEPT  -> Benign Prediction")
    print("    CAUTION -> Attack Prediction")
    print("    REJECT  -> Attack Prediction")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
