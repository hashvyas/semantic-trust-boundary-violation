"""
simulation/llm_dispatcher.py
================================
The rule-based gate sitting between the Trust Decision Engine / DS MASS
and the LLM reasoning agent -- LAST stage of the frozen V2X Trust Stack
(... -> Trust Decision Engine -> Adapters -> DS MASS -> Dispatcher ->
Vehicle Decision). Decides REJECT / IGNORE / EMERGENCY_BRAKE /
FORWARD_TO_LLM.

Integrated from the uploaded modules/decision_engine.py. Renamed per
audit finding D5 -- "decision_engine" is reserved exclusively for
trust_engine/decision_engine.py (the Trust Decision Engine, this
project's actual research contribution). This module performs ZERO
trust fusion; it thresholds an ALREADY-FINAL decision.

ONE FURTHER DUPLICATION FOUND AND RESOLVED DURING THIS INTEGRATION
(not present in the original responsibility audit, found while porting
this specific module): the uploaded decision_engine() directly
re-checked `pki_result["pki_pass"]` and `mbd_result["mbd_pass"]` and
returned "REJECT" itself when either failed. That DUPLICATES trust
fusion logic that trust_engine.decision_engine.TrustDecisionEngine
already owns exclusively (its own "B1 fatal -> REJECT" rule, etc.) --
per the frozen architecture, "No reasoning should exist outside
[the Trust Decision] engine." Fixed: this dispatcher now reads
FinalTrustDecision.trust_level as the SOLE source of REJECT/CAUTION
information -- it never re-inspects pki_result/mbd_result/cp_result
directly. The TTC/emergency-branch/confidence-gate logic below is
genuine downstream vehicle-decision logic (not trust fusion) and is
preserved from the uploaded version, including both bug fixes already
documented there (TTC computed regardless of confidence; explicit
distance_missing handling instead of a silently optimistic default).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


THRESHOLDS = {
    "EMERGENCY_BRAKE": 0.80,
    "AMBULANCE": 0.85,
    "SPAT": 0.90,
    "OBSTACLE": 0.80,
    "ROAD_WORK": 0.70,
}

DEFAULT_THRESHOLD = 0.80

# Event types that bypass the confidence gate and ALWAYS forward to the LLM.
ALWAYS_FORWARD_EVENTS = {"AMBULANCE", "SPAT", "ROAD_WORK"}


def compute_ttc(ego_speed: Optional[float], distance: Optional[float]) -> float:
    """ego_speed in km/h, distance in meters. Returns TTC in seconds.
    inf if ego is stationary or distance is unknown."""
    if ego_speed is None or ego_speed <= 0:
        return float("inf")
    if distance is None:
        return float("inf")
    ego_speed_ms = ego_speed / 3.6
    return distance / ego_speed_ms


def llm_dispatcher(
    final_decision: Any,  # trust_engine.models.FinalTrustDecision
    scene: Dict[str, Any],
    ego_speed: Optional[float] = None,
    ds_mass: Optional[Any] = None,  # adapters.ds_mass_adapter.DSMassOutput, optional
    confidence_gate: bool = True,
    conservative_missing_distance: bool = True,
) -> Dict[str, Any]:
    """
    final_decision: the FinalTrustDecision from TrustDecisionEngine.decide()
        -- the ONLY source of REJECT/trust information this function
        consults. This function does not (and must not) re-inspect
        PKI/B1/MBD/CP internals -- see module docstring.
    scene: dict with at least "event"; may include "distance" (meters)
        and "cp_confidence"/"fusion_confidence" (from the CP layer, via
        the orchestrator's `cp` result).
    ego_speed: this vehicle's own speed in km/h.
    ds_mass: optional DSMassOutput, attached to the result for downstream
        logging/telemetry -- not consulted for the decision itself
        (that's what final_decision.trust_level already is).
    confidence_gate / conservative_missing_distance: see uploaded
        module's original docstring -- both preserved unchanged.
    """
    trust_level = getattr(final_decision, "trust_level", None)
    trust_level_value = getattr(trust_level, "value", trust_level)

    # -----------------------------
    # Trust Decision Engine verdict -- SOLE source of REJECT information.
    # No re-checking of pki_result/mbd_result/cp_result here (see module
    # docstring for the duplication this replaced).
    # -----------------------------
    if trust_level_value == "REJECT":
        return {
            "decision": "REJECT",
            "reason": "TRUST_DECISION_ENGINE_REJECTED",
            "trust_reasoning": getattr(final_decision, "reasoning", None),
            "boundary_stopped_at": "TrustDecisionEngine",
        }

    event = scene.get("event")
    if event is None:
        return {
            "decision": "IGNORE",
            "reason": "NO_EVENT_IN_SCENE",
        }

    confidence = scene.get("cp_confidence", scene.get("fusion_confidence", 1.0))
    threshold = THRESHOLDS.get(event, DEFAULT_THRESHOLD)

    distance_missing = "distance" not in scene or scene.get("distance") is None
    if distance_missing and conservative_missing_distance:
        distance = 15.0
    elif distance_missing:
        distance = 100.0
    else:
        distance = scene["distance"]

    ttc = compute_ttc(ego_speed, distance)

    if confidence_gate and event not in ALWAYS_FORWARD_EVENTS:
        if confidence < threshold:
            return {
                "decision": "IGNORE",
                "reason": "LOW_CONFIDENCE",
                "confidence": confidence,
                "threshold": threshold,
                "ttc": round(ttc, 2) if ttc != float("inf") else None,
                "distance_missing": distance_missing,
            }

    if event == "EMERGENCY_BRAKE" and ttc < 2:
        return {
            "decision": "EMERGENCY_BRAKE",
            "reason": "LOW_TTC",
            "ttc": round(ttc, 2),
            "confidence": confidence,
            "distance_missing": distance_missing,
        }

    if event == "OBSTACLE" and ttc < 3:
        return {
            "decision": "EMERGENCY_BRAKE",
            "reason": "OBSTACLE_AHEAD",
            "ttc": round(ttc, 2),
            "confidence": confidence,
            "distance_missing": distance_missing,
        }

    if event in ("AMBULANCE", "SPAT", "ROAD_WORK"):
        reason_map = {
            "AMBULANCE": "AMBULANCE_EVENT",
            "SPAT": "TRAFFIC_SIGNAL",
            "ROAD_WORK": "ROAD_WORK_EVENT",
        }
        return {
            "decision": "FORWARD_TO_LLM",
            "reason": reason_map[event],
            "ttc": round(ttc, 2) if ttc != float("inf") else None,
            "confidence": confidence,
            "distance_missing": distance_missing,
        }

    return {
        "decision": "FORWARD_TO_LLM",
        "reason": "TRUSTED_EVENT",
        "ttc": round(ttc, 2) if ttc != float("inf") else None,
        "confidence": confidence,
        "distance_missing": distance_missing,
    }
