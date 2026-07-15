"""
validation/run_ablation_table.py
===================================
Roadmap item A4 (the single most important deliverable per the roadmap):
a real, quantitative ablation table -- accuracy/precision/recall/F1,
not just pass/fail counts -- across:

    B1-only | B1+B2 (crypto/structural + MBD/CP-folded explainability) | B1+B2+B3 (full)

for every scenario category with ground-truth labels in this repo
(scenarios/{collusion,fabrication,mixed,replay,sybil}, each message
individually labeled via its "is_attacker" field, plus a benign
negative-control category from test_messages/benign).

Ground truth: message["is_attacker"] (True/False; missing/None treated
as False -- test_messages/benign has no such field, meaning "not an
attacker" by construction of that fixture set).

Prediction: a message is "flagged" if the pipeline's final decision for
that message (run as the target/last message of the growing window, one
step per message in the category, matching how
tests/system_integration_validation.py already processes these same
fixture sets) is CAUTION or REJECT; "not flagged" if ACCEPT.

HONEST LIMITATION (see tests/system_integration_validation.py's own
capability probe for the same caveat): B3's real model requires
torch/GPU. This script probes real availability the same way and labels
its own B1+B2+B3 column accordingly -- "real" if available, or "B3
unavailable (column reflects B1+B2 only; see synthetic Phase-B/flagship
scenarios elsewhere in this repo for B3's fusion-logic behavior when
given real semantic signal)" if not, printed once at the top, not
silently.

Run with: python3 validation/run_ablation_table.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from b1_scsv.scsv import SCSV
from pipeline.orchestrator import ISCEPipeline
from adapters import LoggingAdapter, APIAdapter, DSMassAdapter
from pipeline.b3_bridge import classify_text, preload_classifier

_B3_LOAD_MS = preload_classifier()
_B3_PROBE = classify_text("capability probe")
B3_AVAILABLE = bool(_B3_PROBE.get("available"))


def load_category(name: str) -> List[Dict[str, Any]]:
    d = ROOT / "scenarios" / name
    return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]


def load_benign() -> List[Dict[str, Any]]:
    d = ROOT / "test_messages" / "benign"
    msgs = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
    for m in msgs:
        m.setdefault("is_attacker", False)
    return msgs


CATEGORIES = {
    "Benign (negative control)": load_benign(),
    "Replay attack": load_category("replay"),
    "Sybil attack": load_category("sybil"),
    "Position/content fabrication": load_category("fabrication"),
    "Coordinated collusion": load_category("collusion"),
    "Mixed traffic": load_category("mixed"),
}


def build_pipeline(enable_mbd: bool, enable_cp: bool, b1_only: bool) -> ISCEPipeline:
    scsv = SCSV(cert_rotation_owner="mbd")
    return ISCEPipeline(
        scsv=scsv, enable_mbd=enable_mbd, enable_cp=enable_cp, pki_ca=None,
        adapters={"log": LoggingAdapter(), "api": APIAdapter(), "ds_mass": DSMassAdapter()},
    )


def predict_b1_only(pipe: ISCEPipeline, window: List[Dict[str, Any]]) -> bool:
    """B1-only prediction: bypass B2/B3/fusion entirely -- flag iff B1
    itself reports fatal or non-valid on the target message. This is the
    literal 'B1 in isolation' baseline the roadmap's table asks for."""
    target = window[-1]
    b1_res = pipe.scsv.check_stateful(target)
    valid = getattr(b1_res, "valid", True) if hasattr(b1_res, "valid") else b1_res.get("valid", True)
    fatal = getattr(b1_res, "fatal", False) if hasattr(b1_res, "fatal") else b1_res.get("fatal", False)
    return bool(fatal) or not bool(valid)


def predict_full(pipe: ISCEPipeline, window: List[Dict[str, Any]]) -> str:
    result = pipe.run(list(window), context="urban")
    return result["decision"]  # "ACCEPT" | "CAUTION" | "REJECT"


def confusion_counts(preds: List[bool], truths: List[bool]) -> Tuple[int, int, int, int]:
    tp = sum(1 for p, t in zip(preds, truths) if p and t)
    fp = sum(1 for p, t in zip(preds, truths) if p and not t)
    fn = sum(1 for p, t in zip(preds, truths) if not p and t)
    tn = sum(1 for p, t in zip(preds, truths) if not p and not t)
    return tp, fp, fn, tn


def metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision == precision and recall == recall and (precision + recall) > 0) else float("nan"))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "n": total,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def fmt(v: float) -> str:
    return "n/a" if v != v else f"{v:.3f}"  # v != v checks for NaN


print("=" * 100)
print("ROADMAP A4: QUANTITATIVE ABLATION TABLE (B1-only / B1+B2 / B1+B2+B3)")
print("=" * 100)
print(f"Real B3 model available in this run: {B3_AVAILABLE}")
print("SCORING NOTE (confirmed design intent): CAUTION-until-corroborated for a fresh/")
print("unverified sender (no MBD history yet) is intentional system behavior, not a false")
print("positive. REJECT or CAUTION is scored as 'flagged as attack' against is_attacker ground truth;")
print("CAUTION is reported separately as 'caution_rate' (needs-corroboration rate).")
if not B3_AVAILABLE:
    print("-> B1+B2+B3 column below reflects B1+B2 ONLY (B3 unavailable in this environment,")
    print("   no torch/GPU). B3's actual fusion-logic contribution against real semantic signal")
    print("   is validated separately via the synthetic Phase-B scenarios and flagship scenario")
    print("   elsewhere in this repo -- see tests/test_dempster_shafer_fusion.py and")
    print("   tests/system_integration_validation.py's synthetic sub-tests for that evidence.")
    print("   This table's B1+B2+B3 numbers are therefore a LOWER BOUND on B3's real")
    print("   contribution, not evidence that B3 doesn't help.")
print("=" * 100)

all_rows: Dict[str, Dict[str, Dict[str, float]]] = {}

for cat_name, msgs in CATEGORIES.items():
    print(f"\n--- {cat_name} ({len(msgs)} messages) ---")
    truths = [bool(m.get("is_attacker")) for m in msgs]

    # Column 1: B1-only
    pipe_b1 = build_pipeline(enable_mbd=False, enable_cp=False, b1_only=True)
    preds_b1 = []
    window: List[Dict[str, Any]] = []
    for m in msgs:
        window.append(m)
        preds_b1.append(predict_b1_only(pipe_b1, window))
    m_b1 = metrics(*confusion_counts(preds_b1, truths))

    # Column 2: B1+B2 (MBD+CP folded into B2's evidence, matching this
    # repo's actual architecture -- B3 forced unavailable for this column
    # regardless of real availability, to isolate B1+B2's contribution).
    pipe_b12 = build_pipeline(enable_mbd=True, enable_cp=True, b1_only=False)
    original_classify = sys.modules["pipeline.orchestrator"].classify_text
    sys.modules["pipeline.orchestrator"].classify_text = lambda text, metadata=None: {
        "available": False, "label": None, "confidence": None,
        "risk_level": "unavailable", "status": "FORCED UNAVAILABLE for B1+B2 ablation column",
    }
    decisions_b12 = []
    window = []
    for m in msgs:
        window.append(m)
        decisions_b12.append(predict_full(pipe_b12, window))
    sys.modules["pipeline.orchestrator"].classify_text = original_classify
    # Both REJECT and CAUTION count as flagged/predicted attack.
    preds_b12 = [d in ("REJECT", "CAUTION") for d in decisions_b12]
    caution_rate_b12 = sum(1 for d in decisions_b12 if d == "CAUTION") / len(decisions_b12)
    m_b12 = metrics(*confusion_counts(preds_b12, truths))
    m_b12["caution_rate"] = caution_rate_b12

    # Column 3: B1+B2+B3 (full) -- real B3 if available, else same as
    # column 2 (labeled honestly above/below, not silently).
    pipe_full = build_pipeline(enable_mbd=True, enable_cp=True, b1_only=False)
    decisions_full = []
    window = []
    for m in msgs:
        window.append(m)
        decisions_full.append(predict_full(pipe_full, window))
    preds_full = [d in ("REJECT", "CAUTION") for d in decisions_full]
    caution_rate_full = sum(1 for d in decisions_full if d == "CAUTION") / len(decisions_full)
    m_full = metrics(*confusion_counts(preds_full, truths))
    m_full["caution_rate"] = caution_rate_full

    all_rows[cat_name] = {"B1-only": m_b1, "B1+B2": m_b12, "B1+B2+B3 (full)": m_full}

    for col, m in all_rows[cat_name].items():
        cr = f" caution_rate={fmt(m['caution_rate'])}" if "caution_rate" in m else ""
        print(f"  {col:20s} acc={fmt(m['accuracy'])} prec={fmt(m['precision'])} "
              f"recall={fmt(m['recall'])} f1={fmt(m['f1'])} "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}){cr}")

print("\n" + "=" * 100)
print("MARKDOWN TABLE (paste into paper)")
print("=" * 100)
header = "| Scenario category | B1-only (acc/F1) | B1+B2 (acc/F1/caution%) | B1+B2+B3 (acc/F1/caution%) |"
sep = "|---|---|---|---|"
print(header)
print(sep)
for cat_name, cols in all_rows.items():
    b1 = cols["B1-only"]
    b12 = cols["B1+B2"]
    full = cols["B1+B2+B3 (full)"]
    print(f"| {cat_name} | {fmt(b1['accuracy'])}/{fmt(b1['f1'])} | "
          f"{fmt(b12['accuracy'])}/{fmt(b12['f1'])}/{fmt(b12.get('caution_rate', float('nan')))} | "
          f"{fmt(full['accuracy'])}/{fmt(full['f1'])}/{fmt(full.get('caution_rate', float('nan')))} |")

# Aggregate (micro-averaged) row across all attack categories (excludes
# the benign negative control, which has no positive class by design).
attack_cats = [c for c in all_rows if "Benign" not in c]
for col in ("B1-only", "B1+B2", "B1+B2+B3 (full)"):
    tp = sum(all_rows[c][col]["tp"] for c in attack_cats)
    fp = sum(all_rows[c][col]["fp"] for c in attack_cats)
    fn = sum(all_rows[c][col]["fn"] for c in attack_cats)
    tn = sum(all_rows[c][col]["tn"] for c in attack_cats)
    agg = metrics(tp, fp, fn, tn)
    print(f"\nMicro-averaged across attack categories, {col}: "
          f"acc={fmt(agg['accuracy'])} prec={fmt(agg['precision'])} recall={fmt(agg['recall'])} f1={fmt(agg['f1'])}")

out_path = ROOT / "validation" / "ablation_table_results.json"
out_path.write_text(json.dumps({
    "b3_available": B3_AVAILABLE,
    "results": {cat: {col: m for col, m in cols.items()} for cat, cols in all_rows.items()},
}, indent=2))
print(f"\nFull results written to: {out_path}")
