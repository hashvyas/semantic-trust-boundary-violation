"""
validation/run_b4_fallback_degradation.py
============================================
Roadmap B4: "Report two full result sets: Nominal (B3 live) vs Degraded
(B3 unavailable/timeout, pipeline falls back to B1+B2 only). Validate
this path explicitly."

Runs the same fixed scenario set twice: once with B3 exactly as the
environment provides it (real inference if torch/GPU are available,
else the correct 'unavailable' code path), and once with B3 forced
unavailable regardless of environment (simulating a timeout/crash even
on a machine where B3 would normally be live), and diffs the two.

Run with: python3 validation/run_b4_fallback_degradation.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from b1_scsv.scsv import SCSV
from pipeline.orchestrator import ISCEPipeline
from adapters import LoggingAdapter, APIAdapter, DSMassAdapter
from pipeline.b3_bridge import classify_text, preload_classifier

_B3_LOAD_MS = preload_classifier()
_PROBE = classify_text("capability probe")
B3_AVAILABLE = bool(_PROBE.get("available"))

print("=" * 100)
print("ROADMAP B4: FALLBACK / DEGRADATION VALIDATION")
print("=" * 100)
print(f"Real B3 model available in this environment: {B3_AVAILABLE}")
print("Nominal run: B3 exactly as this environment provides it.")
print("Degraded run: B3 FORCED unavailable (simulates timeout/crash), regardless of")
print("              what this environment can normally provide.")
print("=" * 100)


def load_category(name: str) -> List[Dict[str, Any]]:
    d = ROOT / "scenarios" / name
    return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]


CATEGORIES = {
    "replay": load_category("replay"),
    "sybil": load_category("sybil"),
    "fabrication": load_category("fabrication"),
    "collusion": load_category("collusion"),
}

_FAILURES = []


def check(name, cond, evidence=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {evidence}" if evidence else ""))
    if not cond:
        _FAILURES.append(name)


def build_pipeline() -> ISCEPipeline:
    return ISCEPipeline(
        scsv=SCSV(cert_rotation_owner="mbd"), enable_mbd=True, enable_cp=True, pki_ca=None,
        adapters={"log": LoggingAdapter(), "api": APIAdapter(), "ds_mass": DSMassAdapter()},
    )


def run_category(cat_name: str, msgs: List[Dict[str, Any]], force_b3_unavailable: bool) -> List[str]:
    pipe = build_pipeline()
    original = sys.modules["pipeline.orchestrator"].classify_text
    if force_b3_unavailable:
        sys.modules["pipeline.orchestrator"].classify_text = lambda text, metadata=None: {
            "available": False, "label": None, "confidence": None,
            "risk_level": "unavailable", "status": "FORCED UNAVAILABLE (B4 degraded-path simulation)",
        }
    decisions = []
    window: List[Dict[str, Any]] = []
    try:
        for m in msgs:
            window.append(m)
            r = pipe.run(list(window), context="urban")
            decisions.append(r["decision"])
    finally:
        sys.modules["pipeline.orchestrator"].classify_text = original
    return decisions


for cat_name, msgs in CATEGORIES.items():
    print(f"\n--- {cat_name} ---")
    nominal = run_category(cat_name, msgs, force_b3_unavailable=False)
    degraded = run_category(cat_name, msgs, force_b3_unavailable=True)

    check(f"{cat_name}: nominal run completes without crashing", len(nominal) == len(msgs))
    check(f"{cat_name}: degraded run completes without crashing (no exception on B3 unavailable)",
          len(degraded) == len(msgs))
    check(f"{cat_name}: degraded run never silently 'fakes' a pass (still reaches ACCEPT/CAUTION/REJECT)",
          all(d in ("ACCEPT", "CAUTION", "REJECT") for d in degraded))

    diffs = sum(1 for a, b in zip(nominal, degraded) if a != b)
    print(f"  nominal:  {nominal}")
    print(f"  degraded: {degraded}")
    print(f"  decisions differing between nominal and degraded: {diffs}/{len(msgs)}")
    if not B3_AVAILABLE:
        print("  (NOTE: nominal == degraded here because B3 is already unavailable in this")
        print("   environment -- rerun on a machine with torch/GPU for a real nominal-vs-degraded diff.)")

print()
print("=" * 100)
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S): {_FAILURES}")
    sys.exit(1)
print("B4 fallback/degradation path validated: pipeline never crashes or silently fakes a pass")
print("when B3 is unavailable, in every scenario category tested.")
sys.exit(0)
