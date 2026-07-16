#!/usr/bin/env python3
"""
run_extension_evaluation.py
===========================
Runs run_capped_veremi_test.py on the imported VeReMi Extension datasets
and summarizes the metrics.
"""

from __future__ import annotations
import subprocess
import sys
import re

def run_cmd(args: list[str]) -> str:
    res = subprocess.run(args, capture_output=True, text=True, check=True)
    return res.stdout

def main() -> int:
    datasets = {
        "Constant Position": "data/veremi_ext_constpos.ndjson",
        "Data Replay": "data/veremi_ext_datareplay.ndjson",
        "DoS": "data/veremi_ext_dos.ndjson"
    }

    print("=" * 80)
    print("RUNNING PIPELINE ON VEREMI EXTENSION DATASETS (CAP=200)")
    print("=" * 80)

    for name, path in datasets.items():
        print(f"\nEvaluating attack family: {name} ({path})")
        print("-" * 80)
        
        # Execute capped test
        cmd = [sys.executable, "run_capped_veremi_test.py", "--input", path, "--cap", "200"]
        stdout = run_cmd(cmd)
        
        # Parse metrics from stdout
        # e.g., Accuracy:             0.9500 (95.00%)
        #       Precision:            1.0000 (100.00%)
        #       Recall (Sensitivity):  0.8500 (85.00%)
        #       F1-Score:             0.9189
        
        accuracy = re.search(r"Accuracy:\s+([\d.]+)", stdout)
        precision = re.search(r"Precision:\s+([\d.]+)", stdout)
        recall = re.search(r"Recall \(Sensitivity\):\s+([\d.]+)", stdout)
        f1 = re.search(r"F1-Score:\s+([\d.]+)", stdout)
        
        # Print captured output summary section
        summary_section = stdout.split("EVALUATION METRICS & RESULTS SUMMARY")[-1]
        print(summary_section.strip())
        print("=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
