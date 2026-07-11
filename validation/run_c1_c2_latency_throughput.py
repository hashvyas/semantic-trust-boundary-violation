"""
validation/run_c1_c2_latency_throughput.py
=============================================
Roadmap C1/C2: real end-to-end latency (p50/p95/p99) and throughput
measurement, warm pipeline, per-stage breakdown.

HONEST LIMITATION: this measures every stage's real latency on whatever
hardware runs it, INCLUDING B3's stage IF a real model is loaded in
this environment. If B3 is unavailable (no torch/GPU), B3's own
inference latency cannot be measured here -- bridge_ms will reflect the
(near-zero) cost of the stub/unavailable path, NOT real model inference
time. Rerun this script on a machine with the real B3 model loaded to
get real B3 latency numbers; everything else in this script's numbers
is real regardless.

Run with: python3 validation/run_c1_c2_latency_throughput.py [N_MESSAGES]
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time
from typing import Any, Dict, List

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from b1_scsv.scsv import SCSV
from pipeline.orchestrator import ISCEPipeline
from adapters import LoggingAdapter, APIAdapter, DSMassAdapter
from pipeline.b3_bridge import classify_text, preload_classifier

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300

_B3_LOAD_MS = preload_classifier()
_PROBE = classify_text("capability probe")
B3_AVAILABLE = bool(_PROBE.get("available"))

print("=" * 100)
print("ROADMAP C1/C2: LATENCY (p50/p95/p99) AND THROUGHPUT")
print("=" * 100)
print(f"Real B3 model available in this run: {B3_AVAILABLE}  (load time: {_B3_LOAD_MS:.1f}ms, one-time, excluded below)")
if not B3_AVAILABLE:
    print("-> bridge_ms below reflects the unavailable-path cost only, NOT real B3 inference")
    print("   latency. Rerun on a machine with torch/GPU + the real model loaded for that number.")
print(f"N = {N} messages (repeats the benign fixture set as needed to reach N)")
print("=" * 100)

benign_dir = ROOT / "test_messages" / "benign"
base_msgs = [json.loads(f.read_text()) for f in sorted(benign_dir.glob("*.json"))]
if not base_msgs:
    print("No fixtures found in test_messages/benign -- aborting.")
    sys.exit(1)

pipe = ISCEPipeline(
    scsv=SCSV(cert_rotation_owner="mbd"), enable_mbd=True, enable_cp=True, pki_ca=None,
    adapters={"log": LoggingAdapter(), "api": APIAdapter(), "ds_mass": DSMassAdapter()},
)

# Warm-up (excluded from measurement): absorbs any first-call JIT/caching
# overhead unrelated to steady-state per-message cost.
for _ in range(5):
    pipe.run([base_msgs[0]], context="urban")

stage_samples: Dict[str, List[float]] = {
    "pki_ms": [], "b1_ms": [], "mbd_ms": [], "b2_ms": [], "cp_ms": [],
    "synthesizer_ms": [], "bridge_ms": [], "fusion_ms": [], "total_ms": [],
}

t_throughput_start = time.perf_counter()
for i in range(N):
    m = base_msgs[i % len(base_msgs)]
    r = pipe.run([m], context="urban")
    for k, v in r["latencies"].items():
        stage_samples[k].append(v)
t_throughput_end = time.perf_counter()

wall_s = t_throughput_end - t_throughput_start
throughput_msgs_per_sec = N / wall_s if wall_s > 0 else float("inf")


def pctl(data: List[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


print(f"\nThroughput (sequential, single process): {throughput_msgs_per_sec:.1f} messages/sec "
      f"({N} messages in {wall_s:.3f}s)")
print("\nPer-stage latency (ms):")
print(f"{'stage':16s} {'p50':>8s} {'p95':>8s} {'p99':>8s} {'mean':>8s} {'max':>8s}")
for stage, samples in stage_samples.items():
    if not samples:
        continue
    print(f"{stage:16s} {pctl(samples,0.50):8.3f} {pctl(samples,0.95):8.3f} "
          f"{pctl(samples,0.99):8.3f} {statistics.mean(samples):8.3f} {max(samples):8.3f}")

total_p95 = pctl(stage_samples["total_ms"], 0.95)
budget_ms = 50.0
print(f"\nReal-time budget check: total_ms p95 = {total_p95:.2f}ms vs {budget_ms:.0f}ms target")
print("PASS -- within budget" if total_p95 <= budget_ms else
      f"OVER BUDGET by {total_p95 - budget_ms:.2f}ms "
      f"(bottleneck: {max(stage_samples, key=lambda k: statistics.mean(stage_samples[k]) if stage_samples[k] and k!='total_ms' else -1)})")

out_path = ROOT / "validation" / "latency_throughput_results.json"
out_path.write_text(json.dumps({
    "b3_available": B3_AVAILABLE, "n_messages": N, "throughput_msgs_per_sec": throughput_msgs_per_sec,
    "stage_percentiles_ms": {k: {"p50": pctl(v, 0.5), "p95": pctl(v, 0.95), "p99": pctl(v, 0.99),
                                  "mean": statistics.mean(v) if v else None, "max": max(v) if v else None}
                              for k, v in stage_samples.items()},
}, indent=2))
print(f"\nFull results written to: {out_path}")
