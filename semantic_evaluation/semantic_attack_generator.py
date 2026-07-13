"""
semantic_evaluation/semantic_attack_generator.py
===================================================
Wraps semantic attack payloads (from semantic_attack_dataset.py) into
structurally valid, kinematically plausible V2X CAM messages that will
pass B1 cleanly — ensuring only B3 can detect the malicious content.

The generated messages follow the same JSON schema as the project's
existing test fixtures (see test_messages/benign/normal_car.json).

Key design decision: every message gets plausible, non-anomalous
kinematics so that B1, MBD, and B2 all pass without incident.  The
payload text is injected into ``scene_context.peer_reports`` and/or
``scene_context.rsu_messages``, which are the free-text fields the
synthesizer (pipeline/synthesizer.py) reads and passes to B3.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from semantic_evaluation.semantic_attack_dataset import (
    SemanticAttackScenario,
    ALL_SCENARIOS,
)


# ---------------------------------------------------------------------------
# Physical constants (same scale as scenario_generation/generator.py)
# ---------------------------------------------------------------------------
BASE_LAT = 485512345     # ETSI 1e-7 deg, central Germany
BASE_LON = 96123456
LAT_PER_M = 1e7 / 111132.9
LON_PER_M = 1e7 / (111319.9 * math.cos(math.radians(BASE_LAT * 1e-7)))


def generate_message(
    scenario: SemanticAttackScenario,
    *,
    seed: int = 42,
    station_id: int = 1005,
    station_type: int = 5,        # passengerCar (default — NOT RSU)
    base_timestamp_ms: float = 4200.0,
) -> Dict[str, Any]:
    """Generate a single, structurally valid V2X CAM message carrying
    the semantic attack payload.

    Parameters
    ----------
    scenario : SemanticAttackScenario
        The attack definition (payload text, inject_as target, etc.).
    seed : int
        RNG seed for deterministic kinematic jitter.
    station_id : int
        ETSI station identifier for this vehicle.
    station_type : int
        ETSI station type code (5 = passengerCar).
    base_timestamp_ms : float
        Generation delta time base.

    Returns
    -------
    dict
        A V2X message dict matching the project's JSON schema, with
        metadata fields ``is_attacker``, ``expected_label``,
        ``attack_category``, ``attack_id``.
    """
    rng = random.Random(seed)

    # --- Plausible kinematics (urban context, non-anomalous) ---
    speed_mps = 8.0 + rng.uniform(-2.0, 4.0)   # ~30-45 km/h
    heading_deg10 = 900 + rng.randint(-50, 50)  # roughly east-ish
    yaw_rate = rng.randint(-15, 15)
    lat_offset = rng.uniform(-200, 200) * LAT_PER_M
    lon_offset = rng.uniform(-200, 200) * LON_PER_M
    ts_jitter = rng.uniform(-1.0, 1.0)

    msg: Dict[str, Any] = {
        "header": {
            "station_id": station_id,
            "message_id": 1,
        },
        "cam": {
            "generation_delta_time": round(base_timestamp_ms + ts_jitter, 2),
            "cam_parameters": {
                "basic_container": {
                    "station_type": station_type,
                    "reference_position": {
                        "latitude": int(BASE_LAT + lat_offset),
                        "longitude": int(BASE_LON + lon_offset),
                    },
                },
                "high_frequency_container": {
                    "basic_vehicle_container_high_frequency": {
                        "speed": int(speed_mps * 100),
                        "heading": heading_deg10 % 3600,
                        "yaw_rate": yaw_rate,
                        "steering_wheel_angle": 0,
                        "lateral_acceleration": 0,
                        "longitudinal_acceleration": rng.randint(-20, 30),
                    }
                },
            },
        },
        "certificate_id": f"CERT_{station_type}_{station_id}",
        "cert_id": f"CERT_{station_type}_{station_id}",
        # Local perception: clean (no sensor contradictions)
        "local_perception": {
            "camera": "CLEAR",
            "radar": "CLEAR",
            "lidar": "CLEAR",
        },
        "scene_context": {
            "peer_reports": [],
            "rsu_messages": [],
        },
        # Ground-truth metadata for evaluation
        "is_attacker": scenario.expected_label == "MALICIOUS",
        "expected_label": scenario.expected_label,
        "attack_category": scenario.category,
        "attack_subcategory": scenario.subcategory,
        "attack_id": scenario.id,
        "difficulty": scenario.difficulty,
    }

    # --- Inject the semantic payload into the appropriate free-text field ---
    if scenario.inject_as in ("peer", "both"):
        msg["scene_context"]["peer_reports"].append(scenario.payload_text)
    if scenario.inject_as in ("rsu", "both"):
        msg["scene_context"]["rsu_messages"].append(scenario.payload_text)

    return msg


def generate_corpus(
    scenarios: Optional[List[SemanticAttackScenario]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate V2X messages for the entire corpus (or a subset).

    Returns a list of dicts, one per scenario, deterministically
    ordered and seeded.
    """
    if scenarios is None:
        scenarios = ALL_SCENARIOS

    messages: List[Dict[str, Any]] = []
    for i, scenario in enumerate(scenarios):
        msg = generate_message(
            scenario,
            seed=seed + i,
            station_id=1005 + (i % 50),
            base_timestamp_ms=4200.0 + i * 100.0,
        )
        messages.append(msg)
    return messages


if __name__ == "__main__":
    msgs = generate_corpus()
    n_attack = sum(1 for m in msgs if m["is_attacker"])
    n_benign = sum(1 for m in msgs if not m["is_attacker"])
    print(f"Generated {len(msgs)} messages ({n_attack} attack, {n_benign} benign)")
    # Quick structural sanity
    for m in msgs:
        assert "header" in m
        assert "cam" in m
        assert "scene_context" in m
        assert m["expected_label"] in ("MALICIOUS", "BENIGN")
    print("All messages pass structural sanity check.")
