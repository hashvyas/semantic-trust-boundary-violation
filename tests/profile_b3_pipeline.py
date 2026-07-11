"""
tests/profile_b3_pipeline.py
================================
Evidence-based profiling of the full B3 pipeline: import time, model
loading, GPU validation, and per-message inference -- broken into every
stage requested, each measured independently with time.perf_counter().

Run in a FRESH process:

    python3 tests/profile_b3_pipeline.py

Two specific hypotheses are directly tested (not just asserted) because
of the environment this is running in (WSL2, model path under
/mnt/c/Users/...):

  H1: HuggingFace `from_pretrained()` is making an online Hub API call
      even for a local path, and that network round-trip is slow/timing
      out under WSL2. Tested by re-running model loading a second time
      with local_files_only=True and comparing.

  H2: The model files sit on a Windows-mounted drive (/mnt/c/...), and
      WSL2's 9p filesystem protocol for cross-OS file access is much
      slower than native ext4. Tested by measuring raw disk read
      throughput of the checkpoint file directly, independent of
      torch/transformers entirely.

This script does not optimize anything. It only measures and reports.
"""

from __future__ import annotations

import os
import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STAGE_TIMES = {}


def stage(name):
    """Context manager that times a stage and stores the result."""
    class _Stage:
        def __enter__(self):
            self.t0 = time.perf_counter()
            print(f"  -> starting: {name} ...", flush=True)
            return self

        def __exit__(self, *exc):
            dt = time.perf_counter() - self.t0
            STAGE_TIMES[name] = dt
            print(f"     done: {name} = {dt:.3f}s", flush=True)
            return False
    return _Stage()


print("=" * 70)
print("B3 PIPELINE PROFILING -- fresh process, every stage measured")
print("=" * 70)

t_script_start = time.perf_counter()

# ------------------------------------------------------------------
# 1. Import timing (measured individually, in isolation)
# ------------------------------------------------------------------
with stage("import: yaml"):
    import yaml

with stage("import: transformers"):
    import transformers

with stage("import: torch"):
    import torch

print(f"\n  transformers version: {transformers.__version__}")
print(f"  torch version: {torch.__version__}")

# ------------------------------------------------------------------
# 2. H2 -- raw disk read throughput of the checkpoint file, with NO
#    torch/transformers involvement at all.
# ------------------------------------------------------------------
model_dir = ROOT / "b3" / "solution_stb" / "b3_semantic_gate" / "model" / "semantic_gate_v3"
weights_file = model_dir / "pytorch_model.bin"

print(f"\n  Model directory: {model_dir}")
print(f"  Weights file exists: {weights_file.exists()}")
if weights_file.exists():
    size_mb = weights_file.stat().st_size / (1024 * 1024)
    print(f"  Weights file size: {size_mb:.1f} MB")
    with stage(f"H2 test: raw disk read of pytorch_model.bin ({size_mb:.0f}MB, no torch)"):
        with open(weights_file, "rb") as f:
            total_read = 0
            chunk_size = 64 * 1024 * 1024
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                total_read += len(chunk)
    raw_read_mb_per_s = size_mb / STAGE_TIMES[f"H2 test: raw disk read of pytorch_model.bin ({size_mb:.0f}MB, no torch)"]
    print(f"     raw disk throughput: {raw_read_mb_per_s:.1f} MB/s")
    print(f"     (for reference: native SSD is typically 500-3000 MB/s; "
          f"WSL2 /mnt/c 9p access is commonly 20-100 MB/s)")
else:
    print("  WARNING: weights file not found, skipping H2 disk test.")

# ------------------------------------------------------------------
# 3. Config loading
# ------------------------------------------------------------------
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

with stage("config loading (AutoConfig.from_pretrained)"):
    config = AutoConfig.from_pretrained(str(model_dir))

# ------------------------------------------------------------------
# 4. Tokenizer loading
# ------------------------------------------------------------------
with stage("tokenizer loading (AutoTokenizer.from_pretrained, default -- may hit network)"):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

# ------------------------------------------------------------------
# 5. H1 -- model loading, default (may attempt an online Hub check)
# ------------------------------------------------------------------
with stage("model loading, DEFAULT (local path, from_pretrained default flags)"):
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

# ------------------------------------------------------------------
# 6. H1 direct test -- reload with local_files_only=True explicitly
# ------------------------------------------------------------------
with stage("H1 test: model RE-loading with local_files_only=True (forces no network)"):
    model_local_only = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )
del model_local_only

# ------------------------------------------------------------------
# 7. GPU validation
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("GPU VALIDATION")
print("=" * 70)
cuda_available = torch.cuda.is_available()
print(f"  torch.cuda.is_available(): {cuda_available}")
if cuda_available:
    print(f"  current device index: {torch.cuda.current_device()}")
    print(f"  device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"  allocated memory: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"  reserved memory: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
else:
    print("  No CUDA device visible to torch. Model will run on CPU.")
    print("  (Common under WSL2 if NVIDIA CUDA-on-WSL drivers aren't installed,")
    print("   or if this environment's torch build is CPU-only -- check with:")
    print("   `python3 -c \"import torch; print(torch.__version__)\"` -- a version")
    print("   string WITHOUT '+cuXXX' means a CPU-only torch build is installed.)")

device = torch.device("cuda" if cuda_available else "cpu")
with stage(f"model.to({device})"):
    model = model.to(device)
model.eval()

print(f"\n  model device (from a parameter): {next(model.parameters()).device}")

# ------------------------------------------------------------------
# 8. CUDA context / warm-up inference
# ------------------------------------------------------------------
warmup_text = "V2X Scene Report: context=urban. Ego vehicle: station 1, speed=50."
with stage("warm-up inference (first call -- includes CUDA context creation if GPU present)"):
    with torch.no_grad():
        enc = tokenizer([warmup_text], max_length=256, padding=True, truncation=True, return_tensors="pt").to(device)
        print(f"     input tensor device after .to(device): {enc['input_ids'].device}")
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=1)
        if cuda_available:
            torch.cuda.synchronize()

if cuda_available:
    print(f"  allocated memory after warm-up: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"  reserved memory after warm-up: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

# ------------------------------------------------------------------
# 9. Steady-state per-message inference, broken into sub-stages
# ------------------------------------------------------------------
N = 20
test_text = ("V2X Scene Report: context=rural. Ego vehicle: station 1001 "
             "(type=passengerCar), position=(lat=485512345, lon=96123456), "
             "speed=1500, heading=900 deg. No peer reports received.")

preprocess_times, tokenize_times, inference_times, postprocess_times = [], [], [], []

for i in range(N):
    t0 = time.perf_counter()
    texts = [test_text]
    t1 = time.perf_counter()

    with torch.no_grad():
        enc = tokenizer(texts, max_length=256, padding=True, truncation=True, return_tensors="pt").to(device)
    t2 = time.perf_counter()

    with torch.no_grad():
        out = model(**enc)
        if cuda_available:
            torch.cuda.synchronize()
    t3 = time.perf_counter()

    probs = torch.softmax(out.logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    label = getattr(model.config, "id2label", {}).get(int(preds[0]), "UNKNOWN")
    t4 = time.perf_counter()

    preprocess_times.append(t1 - t0)
    tokenize_times.append(t2 - t1)
    inference_times.append(t3 - t2)
    postprocess_times.append(t4 - t3)

def _avg_ms(lst):
    return (sum(lst) / len(lst)) * 1000.0

print("\n" + "=" * 70)
print(f"STEADY-STATE PER-MESSAGE LATENCY (averaged over {N} calls)")
print("=" * 70)
print(f"  preprocessing:  {_avg_ms(preprocess_times):.3f} ms")
print(f"  tokenization:   {_avg_ms(tokenize_times):.3f} ms")
print(f"  inference:      {_avg_ms(inference_times):.3f} ms")
print(f"  post-processing:{_avg_ms(postprocess_times):.3f} ms")
total_per_msg = _avg_ms(preprocess_times) + _avg_ms(tokenize_times) + _avg_ms(inference_times) + _avg_ms(postprocess_times)
print(f"  TOTAL per-message (measured, this script): {total_per_msg:.3f} ms")

# ------------------------------------------------------------------
# 10. Full report
# ------------------------------------------------------------------
t_script_total = time.perf_counter() - t_script_start

print("\n" + "=" * 70)
print("FULL STAGE REPORT")
print("=" * 70)
print(f"{'Stage':<70} {'Time (s)':>10} {'% of total':>12}")
print("-" * 94)
for name, dt in STAGE_TIMES.items():
    pct = (dt / t_script_total) * 100.0
    print(f"{name:<70} {dt:>10.3f} {pct:>11.1f}%")
print("-" * 94)
print(f"{'TOTAL SCRIPT WALL TIME':<70} {t_script_total:>10.3f} {'100.0':>11}%")

print("\n" + "=" * 70)
print("HYPOTHESIS TEST RESULTS")
print("=" * 70)
default_load = STAGE_TIMES.get("model loading, DEFAULT (local path, from_pretrained default flags)")
local_only_load = STAGE_TIMES.get("H1 test: model RE-loading with local_files_only=True (forces no network)")
if default_load and local_only_load:
    ratio = default_load / local_only_load if local_only_load > 0 else float("inf")
    print(f"  H1 (online Hub check slowing load): default={default_load:.2f}s vs "
          f"local_files_only={local_only_load:.2f}s (ratio {ratio:.1f}x)")
    if ratio > 3:
        print("  -> H1 SUPPORTED by measurement: default loading is significantly")
        print("     slower than forced-local loading. Network/Hub-check overhead")
        print("     is a real contributor. NOTE: local_files_only load benefits from")
        print("     OS file-cache warmth from the first load -- re-run this script")
        print("     twice and compare stage 6 across runs to control for that.")
    else:
        print("  -> H1 NOT supported by measurement: default vs local_files_only")
        print("     load times are comparable. Network/Hub-check is not the")
        print("     dominant cost here.")

disk_key = [k for k in STAGE_TIMES if k.startswith("H2 test")]
if disk_key:
    disk_time = STAGE_TIMES[disk_key[0]]
    print(f"\n  H2 (WSL /mnt/c disk I/O bottleneck): raw read of the checkpoint took "
          f"{disk_time:.2f}s with zero torch/transformers involvement.")
    if disk_time > 5:
        print("  -> H2 SUPPORTED by measurement: raw filesystem read alone accounts")
        print(f"     for {disk_time:.1f}s of the ~{t_script_total:.0f}s total. This is a real,")
        print("     measured filesystem-speed finding, independent of any ML code.")
    else:
        print("  -> H2 NOT supported by measurement: raw disk read is fast; the")
        print("     bottleneck is elsewhere (see stage table above).")

print("\nDone. Copy this ENTIRE output back for root-cause analysis.")