"""
diagnose_b3_latency.py
=======================
TEMPORARY DIAGNOSTIC SCRIPT -- not part of the production pipeline.
Delete after the investigation is done; do not import this from anywhere
in pki/, b1_scsv/, mbd/, cp/, b2_explain/, pipeline/, trust_engine/,
adapters/, or bridges/.

Purpose
-------
Investigate why bridge_ms (B3 inference time) is ~300-1600ms/message
instead of the low-double-digit ms a 142M-param model should take on a
CUDA GPU. Hypothesis under test: varying token/tensor shapes across
messages are forcing repeated cuDNN kernel selection/warmup on WSL2's
GPU passthrough layer, rather than this being a one-time cold-start cost.

Method
------
1. Load REAL messages from test_messages/ and scenarios/ (no synthetic
   English strings) and run them through the actual ISCEPipeline to get
   the actual synthesized_message text B3 sees in production -- this
   reuses pipeline.orchestrator.ISCEPipeline and pipeline.synthesizer
   exactly as-is, it does not reimplement synthesis logic.
2. For each synthesized text, separately time: tokenization, host->GPU
   transfer, forward pass, softmax/argmax post-processing -- using the
   SAME cached predictor singleton the pipeline itself uses (via
   b3.solution_stb.b3_semantic_gate.inference.get_predictor with the
   same model_path/max_length/device the pipeline resolved), so this
   does not construct a second model instance.
3. Runs the full batch TWICE in the same process. If round 2 is much
   faster than round 1 for matching shapes, that supports the
   kernel-selection/warmup hypothesis. If not, we look at CUDA sync
   overhead / host-side Python overhead next.

Nothing in pki/, b1_scsv/, mbd/, cp/, b2_explain/, pipeline/,
trust_engine/, adapters/, bridges/ or isce_config.yaml is modified.
This script only *calls* existing public entry points.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

# Ensure repo root importable when running this script from anywhere.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# 1. Load real message fixtures
# ---------------------------------------------------------------------------

def _load_message_windows(roots: List[str]) -> List[List[Dict[str, Any]]]:
    """Loads every .json fixture under the given roots as a message window.

    A file containing a JSON list is treated as an already-built window
    (target message last, per ISCEPipeline.run's contract). A file
    containing a single JSON object is treated as a single-message
    window. Files that fail to parse are skipped with a printed warning
    rather than crashing the whole diagnostic run.
    """
    windows: List[List[Dict[str, Any]]] = []
    seen_paths = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                print(f"[skip] {path}: failed to parse ({e})")
                continue
            if isinstance(data, list):
                if not data:
                    print(f"[skip] {path}: empty list")
                    continue
                windows.append(data)
            elif isinstance(data, dict):
                windows.append([data])
            else:
                print(f"[skip] {path}: unrecognized JSON shape ({type(data)})")
                continue
            seen_paths.append(path)
    print(f"Loaded {len(windows)} message window(s) from: {seen_paths}")
    return windows


# ---------------------------------------------------------------------------
# 2. Produce the REAL synthesized text via the actual pipeline
# ---------------------------------------------------------------------------

@dataclass
class SynthesizedSample:
    source_window_index: int
    num_messages_in_window: int
    text: str


def _synthesize_all(windows: List[List[Dict[str, Any]]]) -> List[SynthesizedSample]:
    from pipeline.orchestrator import ISCEPipeline

    # enable_mbd/enable_cp default False -- identical to the baseline runs
    # you already captured with manual_pipeline_test.py, so the synthesized
    # text here matches what you've been seeing in bridge_ms measurements.
    pipeline = ISCEPipeline()

    samples: List[SynthesizedSample] = []
    for idx, window in enumerate(windows):
        try:
            result = pipeline.run(window)
        except Exception as e:
            print(f"[skip] window {idx}: pipeline.run failed ({e})")
            continue
        text = result["synthesized_message"]["text"]
        samples.append(SynthesizedSample(
            source_window_index=idx,
            num_messages_in_window=len(window),
            text=text,
        ))
    return samples


# ---------------------------------------------------------------------------
# 3. Stage-by-stage B3 timing, reusing the pipeline's own cached predictor
# ---------------------------------------------------------------------------

@dataclass
class StageTiming:
    sample_index: int
    round_num: int
    char_len: int
    token_count: int
    tensor_shape: tuple
    preprocessing_ms: float
    tokenization_ms: float
    gpu_transfer_ms: float
    inference_ms: float
    postprocessing_ms: float
    total_ms: float
    label: str
    confidence: float


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_one(predictor, text: str, sample_index: int, round_num: int) -> StageTiming:
    device = predictor.device

    # --- preprocessing: whatever happens before the tokenizer touches the
    # text. In the current implementation this is effectively nothing
    # (predict() passes the string straight to the tokenizer), but we time
    # it explicitly rather than assume, per the "measure, don't assume" rule.
    t0 = time.perf_counter()
    batch = [text]
    _sync_if_cuda(device)
    t1 = time.perf_counter()
    preprocessing_ms = (t1 - t0) * 1000.0

    # --- tokenization (CPU-side)
    enc_cpu = predictor.tokenizer(
        batch,
        max_length=predictor.max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    _sync_if_cuda(device)
    t2 = time.perf_counter()
    tokenization_ms = (t2 - t1) * 1000.0

    token_count = int(enc_cpu["input_ids"].shape[1])
    tensor_shape = tuple(enc_cpu["input_ids"].shape)

    # --- host -> GPU transfer
    enc = enc_cpu.to(device)
    _sync_if_cuda(device)
    t3 = time.perf_counter()
    gpu_transfer_ms = (t3 - t2) * 1000.0

    # --- forward pass
    with torch.no_grad():
        out = predictor.model(**enc)
    _sync_if_cuda(device)
    t4 = time.perf_counter()
    inference_ms = (t4 - t3) * 1000.0

    # --- post-processing (softmax/argmax/label lookup, mirrors predict())
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()
    pred = int(probs.argmax(axis=1)[0])
    conf = float(probs.max(axis=1)[0])
    label = predictor.id2label.get(pred, f"LABEL_{pred}")
    _sync_if_cuda(device)
    t5 = time.perf_counter()
    postprocessing_ms = (t5 - t4) * 1000.0

    total_ms = (t5 - t0) * 1000.0

    return StageTiming(
        sample_index=sample_index,
        round_num=round_num,
        char_len=len(text),
        token_count=token_count,
        tensor_shape=tensor_shape,
        preprocessing_ms=preprocessing_ms,
        tokenization_ms=tokenization_ms,
        gpu_transfer_ms=gpu_transfer_ms,
        inference_ms=inference_ms,
        postprocessing_ms=postprocessing_ms,
        total_ms=total_ms,
        label=label,
        confidence=conf,
    )


# ---------------------------------------------------------------------------
# 4. Environment info
# ---------------------------------------------------------------------------

def _print_environment(predictor) -> None:
    import transformers
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    print(f"torch.__version__:            {torch.__version__}")
    print(f"transformers.__version__:     {transformers.__version__}")
    print(f"torch.version.cuda:           {torch.version.cuda}")
    print(f"torch.cuda.is_available():    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device:                    {torch.cuda.get_device_name(0)}")
    print(f"torch.backends.cudnn.enabled:   {torch.backends.cudnn.enabled}")
    print(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"predictor.device:               {predictor.device}")
    print(f"model num params:               {sum(p.numel() for p in predictor.model.parameters())}")
    print()


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> None:
    from b3.solution_stb.b3_semantic_gate.inference import get_predictor
    from pipeline.b3_bridge import SemanticGateClassifier, _load_b3_config

    roots = ["test_messages", "scenarios"]
    windows = _load_message_windows(roots)
    if not windows:
        print("No message fixtures found under test_messages/ or scenarios/. Aborting.")
        return

    samples = _synthesize_all(windows)
    if not samples:
        print("No synthesized messages produced (all windows failed pipeline.run). Aborting.")
        return

    # Reuse the exact model_path/max_length/device the pipeline resolved,
    # so get_predictor() returns the SAME cached singleton instance the
    # pipeline itself is using -- not a second model load.
    b3_config = _load_b3_config()
    model_path = b3_config.get("model_path", "b3/solution_stb/b3_semantic_gate/model/semantic_gate_v3")
    max_length = b3_config.get("max_length", 256)
    device_cfg = b3_config.get("device", None)
    predictor = get_predictor(model_path, max_length=max_length, device=device_cfg)

    _print_environment(predictor)

    print(f"Profiling {len(samples)} real synthesized message(s), 2 rounds each.\n")

    all_timings: List[StageTiming] = []
    for round_num in (1, 2):
        print("-" * 70)
        print(f"ROUND {round_num}")
        print("-" * 70)
        for s in samples:
            timing = _time_one(predictor, s.text, s.source_window_index, round_num)
            all_timings.append(timing)
            print(
                f"[R{round_num}] window={s.source_window_index:>3} "
                f"chars={timing.char_len:>4} tokens={timing.token_count:>4} "
                f"shape={timing.tensor_shape} "
                f"preproc={timing.preprocessing_ms:6.2f}ms "
                f"tok={timing.tokenization_ms:6.2f}ms "
                f"xfer={timing.gpu_transfer_ms:6.2f}ms "
                f"infer={timing.inference_ms:7.2f}ms "
                f"post={timing.postprocessing_ms:5.2f}ms "
                f"TOTAL={timing.total_ms:7.2f}ms "
                f"label={timing.label} conf={timing.confidence:.4f}"
            )
        print()

    # -----------------------------------------------------------------
    # 6. Round 1 vs Round 2 comparison
    # -----------------------------------------------------------------
    print("=" * 70)
    print("ROUND 1 vs ROUND 2 COMPARISON (matched by window index)")
    print("=" * 70)

    by_key = {(t.round_num, t.sample_index): t for t in all_timings}
    round1_totals, round2_totals = [], []
    round1_infer, round2_infer = [], []
    shape_mismatch_flagged = False

    for s in samples:
        idx = s.source_window_index
        r1 = by_key.get((1, idx))
        r2 = by_key.get((2, idx))
        if r1 is None or r2 is None:
            continue
        if r1.tensor_shape != r2.tensor_shape:
            shape_mismatch_flagged = True
        speedup_total = (r1.total_ms / r2.total_ms) if r2.total_ms > 0 else float("nan")
        speedup_infer = (r1.inference_ms / r2.inference_ms) if r2.inference_ms > 0 else float("nan")
        print(
            f"window={idx:>3} shape={r1.tensor_shape} "
            f"total: {r1.total_ms:7.2f}ms -> {r2.total_ms:7.2f}ms ({speedup_total:5.2f}x) | "
            f"infer: {r1.inference_ms:7.2f}ms -> {r2.inference_ms:7.2f}ms ({speedup_infer:5.2f}x)"
        )
        round1_totals.append(r1.total_ms)
        round2_totals.append(r2.total_ms)
        round1_infer.append(r1.inference_ms)
        round2_infer.append(r2.inference_ms)

    if shape_mismatch_flagged:
        print(
            "\n[WARN] Tensor shapes differed between round 1 and round 2 for at "
            "least one window -- this can happen if padding=True pads to the "
            "batch's own longest sequence and batch composition differs. "
            "Interpret the per-window comparison above with that in mind; the "
            "aggregate verdict below is still informative."
        )

    if round1_totals and round2_totals:
        avg1 = sum(round1_totals) / len(round1_totals)
        avg2 = sum(round2_totals) / len(round2_totals)
        avg1_inf = sum(round1_infer) / len(round1_infer)
        avg2_inf = sum(round2_infer) / len(round2_infer)
        print(f"\nAverage TOTAL   round 1: {avg1:8.2f}ms   round 2: {avg2:8.2f}ms")
        print(f"Average INFER   round 1: {avg1_inf:8.2f}ms   round 2: {avg2_inf:8.2f}ms")

        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)
        if avg2 < avg1 * 0.5:
            print(
                "Round 2 is more than 2x faster than round 1 on average.\n"
                "SUPPORTS the hypothesis: new tensor shapes are triggering repeated\n"
                "CUDA kernel selection/warmup (cuDNN autotuning and/or WSL2 GPU\n"
                "passthrough overhead), not steady-state per-message cost."
            )
        elif avg2 < avg1 * 0.85:
            print(
                "Round 2 is somewhat faster than round 1, but not dramatically so.\n"
                "PARTIALLY supports the hypothesis -- some warmup effect is present,\n"
                "but it does not fully explain the magnitude seen in bridge_ms.\n"
                "Look at gpu_transfer_ms and tokenization_ms columns above for a\n"
                "secondary contributor (e.g. host<->device transfer overhead under\n"
                "WSL2, or CPU-side tokenization cost) before concluding."
            )
        else:
            print(
                "Round 2 is NOT meaningfully faster than round 1.\n"
                "DOES NOT support the kernel-warmup hypothesis. Look at the stage\n"
                "breakdown above: if gpu_transfer_ms or tokenization_ms dominate\n"
                "consistently across both rounds (rather than inference_ms), the\n"
                "bottleneck is host-side (WSL2 host<->GPU transfer, or CPU\n"
                "tokenization), not CUDA kernel selection, and needs a different\n"
                "fix (e.g. pinned memory, batching multiple messages together, or\n"
                "profiling with torch.profiler for a precise breakdown)."
            )


if __name__ == "__main__":
    main()