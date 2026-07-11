"""
diagnose_b3_warmup_curve.py
=============================
TEMPORARY DIAGNOSTIC SCRIPT -- not part of the production pipeline.
Delete after use. Does not modify any production code.

Purpose
-------
diagnose_b3_latency.py already confirmed that B3 inference latency on
this WSL2/CUDA setup converges gradually over MANY calls (not 1-4), and
that the eventual floor is ~20-30ms. This script finds the actual call
count where convergence happens, using real synthesized pipeline text
(cycled, since we need N calls but only have a handful of fixtures),
so preload_classifier()'s warmup count can be chosen from evidence
instead of guessed.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

N_CALLS = 60


def _load_message_windows(roots: List[str]) -> List[List[Dict[str, Any]]]:
    windows: List[List[Dict[str, Any]]] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            if isinstance(data, list) and data:
                windows.append(data)
            elif isinstance(data, dict):
                windows.append([data])
    return windows


def main() -> None:
    from pipeline.orchestrator import ISCEPipeline
    from pipeline.b3_bridge import _load_b3_config
    from b3.solution_stb.b3_semantic_gate.inference import get_predictor

    windows = _load_message_windows(["test_messages", "scenarios"])
    if not windows:
        print("No fixtures found. Aborting.")
        return

    # IMPORTANT CHANGE from the previous version of this script: last run
    # showed call #1 already at the noise floor (no descent visible at
    # all), because pipeline.run() had already invoked B3 twenty times
    # during synthesized-text extraction, before the timing loop started.
    # That hid the actual convergence point. This version separates
    # "extract text" from "run the model" -- synthesized text is pulled
    # via pipeline.synthesizer.synthesize_message() directly (pure text
    # assembly, does not touch B3), so predictor.predict() is called for
    # the FIRST time inside the timed loop below, from a genuinely cold
    # predictor -- matching a real fresh-process startup.
    from pipeline.synthesizer import synthesize_message
    from b1_scsv.scsv import SCSV
    from b2_explain.explainability import ExplainabilityEngine

    scsv = SCSV()
    b2 = ExplainabilityEngine()

    texts = []
    for w in windows[:20]:
        try:
            target_msg = w[-1]
            b1_res = scsv.check_stateful(target_msg)
            b1_dict = {
                "valid": getattr(b1_res, "valid", True),
                "fatal": getattr(b1_res, "fatal", False),
                "score": getattr(b1_res, "validation_score", getattr(b1_res, "score", 1.0)),
                "confidence": getattr(b1_res, "confidence", 1.0),
                "reasons": getattr(b1_res, "reasons", []),
                "checks": getattr(b1_res, "checks", {}),
                "details": getattr(b1_res, "details", {}),
            }
            b2_report = b2.explain(b1_dict)
            synthesized = synthesize_message(w, b2_report.to_dict(), None)
            texts.append(synthesized["text"])
        except Exception as e:
            print(f"[skip] text extraction failed: {e}")
            continue
    if not texts:
        print("No synthesized text produced. Aborting.")
        return

    b3_config = _load_b3_config()
    model_path = b3_config.get("model_path", "b3/solution_stb/b3_semantic_gate/model/semantic_gate_v3")
    max_length = b3_config.get("max_length", 256)
    device_cfg = b3_config.get("device", None)
    predictor = get_predictor(model_path, max_length=max_length, device=device_cfg)
    device = predictor.device

    print(f"Cycling {len(texts)} real synthesized message(s) for {N_CALLS} calls.")
    print("This predictor has NOT been called yet -- call #1 below is the "
          "true first inference call against this process.")
    print()

    latencies = []
    for i in range(N_CALLS):
        text = texts[i % len(texts)]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictor.predict([text])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        latencies.append(ms)
        print(f"call {i+1:>3}: {ms:7.2f}ms")

    # Find the first call after which latency stays under 1.5x the final
    # 10-call average for the rest of the run -- a simple, inspectable
    # convergence-point heuristic, not a magic constant.
    floor = sum(latencies[-10:]) / 10
    threshold = floor * 1.5
    convergence_call = None
    for i in range(len(latencies)):
        if all(l <= threshold for l in latencies[i:]):
            convergence_call = i + 1
            break

    print()
    print("=" * 70)
    print(f"Floor (avg of last 10 calls): {floor:.2f}ms")
    print(f"Convergence threshold (1.5x floor): {threshold:.2f}ms")
    print(f"First call after which latency stays under threshold: "
          f"{convergence_call if convergence_call else 'not reached within ' + str(N_CALLS)}")
    print("=" * 70)


if __name__ == "__main__":
    main()