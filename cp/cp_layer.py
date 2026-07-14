"""
cp/cp_layer.py
================
Cooperative Perception (CP) layer -- FIFTH stage of the frozen V2X Trust
Stack (PKI -> B1 -> MBD -> B2 -> CP -> B3 -> TrustDecisionEngine ->
Adapters -> DS MASS -> Dispatcher).

Integrated from modules/cp_layer.py ONLY. modules/cp.py (an earlier,
strict-subset draft of this same module -- confirmed via diff during the
responsibility audit, finding D4) is intentionally NOT carried into this
repository. Do not resurrect it; cp_layer.py is the single source of
truth for CP.

Structural change made during integration (algorithm logic is otherwise
byte-for-byte identical to the uploaded version):

1. `boundary` renamed from `"B3_CP"` to `"CP"` -- resolves the label
   collision identified in the responsibility audit (§0). "B3" is
   reserved exclusively for this repo's validated Semantic Trust layer.

Operates on the FLAT {sender, x, y, speed, heading, timestamp} schema
produced by bridges.message_adapter.to_flat_report() -- see that
module's docstring for why this conversion is load-bearing.

`observation_weights` is supplied by B2's ExplainabilityReport (via the
orchestrator), per the audit's confirmed B2-before-CP ordering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def weighted_std(values, weights) -> float:
    weights = np.array(weights, dtype=float)
    values = np.array(values, dtype=float)
    if np.sum(weights) == 0:
        return float(np.std(values))
    weighted_mean = np.average(values, weights=weights)
    variance = np.average((values - weighted_mean) ** 2, weights=weights)
    return float(np.sqrt(variance))


def spatial_consistency(reports, weights=None) -> float:
    xs = [r["x"] for r in reports]
    ys = [r["y"] for r in reports]
    if weights is not None:
        spread = weighted_std(xs, weights) + weighted_std(ys, weights)
    else:
        spread = float(np.std(xs)) + float(np.std(ys))
    score = max(0, 1 - spread / 20)
    return round(score, 3)


def speed_consistency(reports, weights=None) -> float:
    speeds = [r["speed"] for r in reports]
    if weights is not None:
        spread = weighted_std(speeds, weights)
    else:
        spread = float(np.std(speeds))
    score = max(0, 1 - spread / 20)
    return round(score, 3)


def heading_consistency(reports, weights=None) -> float:
    headings = [r["heading"] for r in reports]
    if weights is not None:
        spread = weighted_std(headings, weights)
    else:
        spread = float(np.std(headings))
    score = max(0, 1 - spread / 30)
    return round(score, 3)


def source_diversity(reports, weights_dict=None) -> float:
    if weights_dict is None:
        senders = set(r["sender"] for r in reports)
        return min(1.0, len(senders) / 5)

    unique_senders: Dict[Any, float] = {}
    for r in reports:
        sender = r["sender"]
        weight_info = weights_dict.get(sender, 1.0)
        weight = weight_info.get("weight", 1.0) if isinstance(weight_info, dict) else weight_info
        unique_senders[sender] = max(unique_senders.get(sender, 0.0), weight)

    weighted_sum = sum(unique_senders.values())
    return min(1.0, weighted_sum / 5)


def cp_layer(
    reports: List[Dict[str, Any]],
    event_label: Optional[str] = None,
    observation_weights: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    """reports: list of flat report dicts that already passed PKI/B1/MBD
    individually. CP fuses them into one situational confidence score.
    observation_weights: sender_id -> float or {"weight": float}, from B2.
    """
    # Detect if actual cooperative perception observations of a single target are available
    # If the reports are just self-broadcasts (e.g. from VeReMi logs), CP observations are unavailable.
    observations_available = (event_label is not None) and not any(r.get("source") == "veremi" for r in reports)

    if len(reports) == 0:
        return {
            "boundary": "CP",  # renamed from "B3_CP" -- see module docstring
            "event_label": event_label,
            "num_reports": 0,
            "senders": [],
            "spatial_score": 0.0,
            "speed_score": 0.0,
            "heading_score": 0.0,
            "diversity_score": 0.0,
            "cp_confidence": 0.0,
            "fusion_confidence": 0.0,
            "cp_pass": False,
            "reports": [],
            "observations_available": observations_available,
        }

    if observation_weights is not None:
        weights = []
        for r in reports:
            sender = r["sender"]
            weight_info = observation_weights.get(sender, 1.0)
            weight = weight_info.get("weight", 1.0) if isinstance(weight_info, dict) else weight_info
            weights.append(weight)
    else:
        weights = None

    if not observations_available:
        # CP observations are unavailable -> propagate neutral/vacuous values
        spatial_score = 1.0
        speed_score = 1.0
        heading_score = 1.0
        diversity_score = 1.0
        confidence = 1.0
    else:
        spatial_score = spatial_consistency(reports, weights=weights)
        speed_score = speed_consistency(reports, weights=weights)
        heading_score = heading_consistency(reports, weights=weights)
        diversity_score = source_diversity(reports, weights_dict=observation_weights)

        confidence = (
            spatial_score * 0.35
            + speed_score * 0.25
            + heading_score * 0.20
            + diversity_score * 0.20
        )
        confidence = round(float(confidence), 3)

    return {
        "boundary": "CP",
        "event_label": event_label,
        "num_reports": len(reports),
        "senders": sorted({r["sender"] for r in reports}),
        "spatial_score": float(spatial_score),
        "speed_score": float(speed_score),
        "heading_score": float(heading_score),
        "diversity_score": float(diversity_score),
        "cp_confidence": confidence,
        "fusion_confidence": confidence,
        "cp_pass": bool(confidence > 0.7) if observations_available else True,
        "reports": reports,
        "observations_available": observations_available,
    }
