"""
mbd/mbd_layer.py
==================
Misbehavior Detection (MBD) layer -- THIRD stage of the frozen V2X Trust
Stack (PKI -> B1 -> MBD -> B2 -> CP -> B3 -> TrustDecisionEngine ->
Adapters -> DS MASS -> Dispatcher).

Integrated from the uploaded modules/mbd.py. Structural changes made
during integration (algorithm logic is otherwise byte-for-byte identical
to the uploaded version):

1. `boundary` renamed from `"B2_MBD"` to `"MBD"` -- resolves the label
   collision identified in the responsibility audit (§0). "B2" is
   reserved exclusively for this repo's validated Explainability layer.
2. Operates on the FLAT {sender, x, y, speed, heading, timestamp, event}
   schema produced by bridges.message_adapter.to_flat_report() -- NOT
   raw ETSI CAM messages. See bridges/message_adapter.py's docstring for
   why this conversion is load-bearing (proximity thresholds are in
   meters; feeding raw lat/lon would silently corrupt every score).
3. Per responsibility-audit finding D1, MBD is now the single source of
   truth for certificate-ROTATION-RATE anomaly scoring (a behavioral
   signal -- "is this cert rotating suspiciously often" -- as opposed to
   PKI's cryptographic validity facts). Rather than reimplementing that
   algorithm a second time, MBD DELEGATES to B1's own existing, already-
   tested VehicleStateManager.check_cert_rotation() -- see
   certificate_rotation_score() below. B1 stops applying a score penalty
   for it (see b1_scsv/scsv.py's `cert_rotation_owner` parameter) once
   this is wired in the orchestrator, but the tracking algorithm itself
   lives in exactly one place, per the audit's "single source of truth"
   requirement.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional


MAX_HISTORY = 5
MAX_ACCEL = 6.0
MAX_SPEED_KMH = 180.0
MAX_HEADING_RATE = 90.0


class MBDResult(dict):
    """Structured behavioral evidence produced by MBD. Inherits from dict
    for backward compatibility with the uploaded module's own callers."""

    def __init__(
        self,
        passed: bool,
        behavior_evidence_quality: float,
        kinematic_score: float,
        temporal_consistency: float,
        replay_score: float,
        sybil_score: float,
        collusion_score: float,
        anomaly_score: float,
        evidence: list,
        certificate_rotation_score: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self["passed"] = passed
        self["mbd_pass"] = passed
        self["behavior_evidence_quality"] = behavior_evidence_quality
        self["behavior_confidence"] = behavior_evidence_quality  # deprecated alias
        self["kinematic_score"] = kinematic_score
        self["temporal_consistency"] = temporal_consistency
        self["replay_score"] = replay_score
        self["sybil_score"] = sybil_score
        self["collusion_score"] = collusion_score
        self["anomaly_score"] = anomaly_score
        self["evidence"] = evidence
        self["certificate_rotation_score"] = certificate_rotation_score
        for k, v in kwargs.items():
            self[k] = v

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class VehicleHistoryStore:
    """Keeps a short rolling history of flat reports per sender_id."""

    def __init__(self, max_history: int = MAX_HISTORY) -> None:
        self._history: Dict[Any, deque] = defaultdict(lambda: deque(maxlen=max_history))

    def get(self, sender_id):
        return self._history[sender_id]

    def push(self, sender_id, report) -> None:
        self._history[sender_id].append(report)


def speed_check(msg: Dict[str, Any]) -> bool:
    return 0 <= msg["speed"] <= MAX_SPEED_KMH


def timestamp_check(msg: Dict[str, Any], max_age_sec: float = 5, history_store: Optional[VehicleHistoryStore] = None) -> bool:
    ts = msg.get("timestamp", 0)
    is_offline = 1000 <= ts <= 1_000_000_000
    if is_offline:
        if history_store is not None:
            max_ts = max((h[-1]["timestamp"] for h in history_store._history.values() if h), default=ts)
            ref_time = max(max_ts, ts)
        else:
            ref_time = ts
        return (ref_time - ts) <= max_age_sec * 1000.0
    else:
        if ts < 1000:
            return True
        if ts > 1_000_000_000:
            return True
        return abs(ts - time.time()) <= max_age_sec


def position_check(msg: Dict[str, Any], bound: float = 100_000) -> bool:
    return abs(msg["x"]) < bound and abs(msg["y"]) < bound


def heading_check(msg: Dict[str, Any]) -> bool:
    return 0 <= msg["heading"] <= 360


def kinematic_consistency_check(msg: Dict[str, Any], history: deque):
    if len(history) == 0:
        return True, {"reason": "no_history_yet"}

    prev = history[-1]
    dt = msg["timestamp"] - prev["timestamp"]
    if dt <= 0:
        return False, {"reason": "non_increasing_timestamp"}

    dx = msg["x"] - prev["x"]
    dy = msg["y"] - prev["y"]
    implied_speed_ms = ((dx ** 2 + dy ** 2) ** 0.5) / dt
    implied_speed_kmh = implied_speed_ms * 3.6

    reported_speed_prev_ms = prev["speed"] / 3.6
    reported_speed_now_ms = msg["speed"] / 3.6
    accel = abs(reported_speed_now_ms - reported_speed_prev_ms) / dt

    heading_delta = abs(msg["heading"] - prev["heading"]) % 360
    heading_delta = min(heading_delta, 360 - heading_delta)
    heading_rate = heading_delta / dt

    speed_jump_ok = abs(implied_speed_kmh - msg["speed"]) <= 40
    accel_ok = accel <= MAX_ACCEL
    heading_rate_ok = heading_rate <= MAX_HEADING_RATE

    is_consistent = speed_jump_ok and accel_ok and heading_rate_ok
    detail = {
        "dt": round(dt, 3),
        "implied_speed_kmh": round(implied_speed_kmh, 2),
        "accel_ms2": round(accel, 2),
        "heading_rate_dps": round(heading_rate, 2),
        "speed_jump_ok": speed_jump_ok,
        "accel_ok": accel_ok,
        "heading_rate_ok": heading_rate_ok,
    }
    return is_consistent, detail


def certificate_rotation_score(
    is_cert_anomaly: Optional[bool],
) -> Optional[float]:
    """Folds B1's VehicleStateManager.check_cert_rotation() boolean verdict
    (the single source of truth for this algorithm, per audit finding D1)
    into MBD's [0,1] score convention. Returns None if the signal wasn't
    supplied (e.g. B1 not wired with cert-rotation delegation enabled),
    so callers can distinguish "no rotation anomaly" from "signal
    unavailable" rather than conflating them.
    """
    if is_cert_anomaly is None:
        return None
    return 1.0 if is_cert_anomaly else 0.0


def mbd_layer(
    msg: Dict[str, Any],
    history_store: VehicleHistoryStore,
    cert_rotation_anomaly: Optional[bool] = None,
) -> MBDResult:
    """Analyzes a FLAT report (see bridges.message_adapter.to_flat_report)
    relative to physical limits and history.

    cert_rotation_anomaly: optional bool from B1's VehicleStateManager,
        passed through by the orchestrator when cert_rotation_owner="mbd"
        (see module docstring, point 3). None if not supplied.
    """
    sender_id = msg["sender"]
    history = history_store.get(sender_id)

    plausibility = {
        "speed_valid": speed_check(msg),
        "timestamp_valid": timestamp_check(msg, history_store=history_store),
        "position_valid": position_check(msg),
        "heading_valid": heading_check(msg),
    }
    plausibility_pass = all(plausibility.values())

    evidence = []
    if not plausibility["speed_valid"]:
        evidence.append(f"Speed out of bounds: {msg['speed']} km/h (limit: [0, {MAX_SPEED_KMH}])")
    if not plausibility["timestamp_valid"]:
        evidence.append(f"Timestamp invalid or too old: {msg.get('timestamp')}")
    if not plausibility["position_valid"]:
        evidence.append(f"Position out of bounds: x={msg['x']}, y={msg['y']}")
    if not plausibility["heading_valid"]:
        evidence.append(f"Heading out of bounds: {msg['heading']} deg (limit: [0, 360])")

    kinematics_ok, kinematics_detail = kinematic_consistency_check(msg, history)

    if len(history) == 0:
        kinematic_score = 1.0
        temporal_consistency = 1.0
        evidence.append("No history yet; baseline assumed valid.")
    else:
        speed_jump_score = 1.0 if kinematics_detail.get("speed_jump_ok", True) else 0.0
        accel = kinematics_detail.get("accel_ms2", 0.0)
        accel_score = max(0.0, 1.0 - (accel / MAX_ACCEL))
        if not kinematics_detail.get("accel_ok", True):
            evidence.append(f"Acceleration {accel:.2f} m/s^2 exceeds threshold {MAX_ACCEL} m/s^2.")
        heading_rate = kinematics_detail.get("heading_rate_dps", 0.0)
        heading_rate_score = max(0.0, 1.0 - (heading_rate / MAX_HEADING_RATE))
        if not kinematics_detail.get("heading_rate_ok", True):
            evidence.append(f"Heading rate {heading_rate:.2f} deg/s exceeds threshold {MAX_HEADING_RATE} deg/s.")

        kinematic_score = round((speed_jump_score + accel_score + heading_rate_score) / 3.0, 3)

        dt = kinematics_detail.get("dt", 1.0)
        if dt <= 0:
            temporal_consistency = 0.0
            evidence.append(f"Chronology violation: timestamp delta {dt} is non-positive.")
        else:
            ts = msg.get("timestamp", 0)
            is_offline = 1000 <= ts <= 1_000_000_000
            if is_offline:
                max_ts = max((h[-1]["timestamp"] for h in history_store._history.values() if h), default=ts)
                age = max_ts - ts
                age_score = max(0.0, 1.0 - age / 5000.0)
            else:
                if ts < 1000 or ts > 1_000_000_000:
                    age_score = 1.0
                else:
                    age = abs(ts - time.time())
                    age_score = max(0.0, 1.0 - age / 5.0)
            temporal_consistency = round(age_score, 3)

    # Replay check (Fix 3)
    replay_score = 0.0
    ts_now = msg.get("timestamp", 0)
    msg_id = msg.get("messageID")
    
    # Check historical sender state for duplicate, delayed, and non-monotonic timestamps
    if len(history) > 0:
        prev_msg = history[-1]
        prev_ts = prev_msg.get("timestamp", 0)
        
        # 1. Non-monotonic timestamp
        if ts_now < prev_ts:
            replay_score = 1.0
            evidence.append(f"Replay check: Non-monotonic timestamp detected (current {ts_now} < previous {prev_ts}).")
        
        # 2. Duplicate timestamp + message ID
        elif ts_now == prev_ts:
            if msg_id is not None and prev_msg.get("messageID") == msg_id:
                replay_score = 1.0
                evidence.append(f"Replay check: Duplicate message ID {msg_id} and timestamp {ts_now} detected.")
            else:
                replay_score = 1.0
                evidence.append(f"Replay check: Duplicate message timestamp {ts_now} detected.")
        
        # 3. Delayed packet relative to sender history
        elif prev_ts - ts_now > 5.0:
            replay_score = 1.0
            evidence.append(f"Replay check: Delayed packet received (delay of {prev_ts - ts_now:.1f}s relative to sender history).")

        # 3.1. Constant Position Attack Check (Fix 2)
        # If the vehicle reports moving (reported speed > 5 km/h) but its position remains constant over consecutive messages.
        # We check cumulative displacement from the first message in our history window.
        if replay_score < 1.0:
            first_msg = history[0]
            cumulative_dt = ts_now - first_msg.get("timestamp", 0)
            if cumulative_dt >= 1.0:
                total_dx = msg["x"] - first_msg["x"]
                total_dy = msg["y"] - first_msg["y"]
                total_dist = (total_dx**2 + total_dy**2)**0.5
                
                # Calculate average reported speed in km/h across history
                avg_reported_speed = sum(h["speed"] for h in history) / len(history)
                
                # If displacement is tiny (< 1.0 meter) but average reported speed indicates significant motion (> 5.0 km/h)
                if total_dist < 1.0 and avg_reported_speed > 5.0:
                    replay_score = 1.0
                    evidence.append(f"Constant Position check: Vehicle reports movement (avg speed {avg_reported_speed:.2f} km/h) but displacement is only {total_dist:.2f}m over {cumulative_dt:.2f}s.")

    # 4. Duplicate message ID and timestamp check across all history for this sender
    if replay_score < 1.0 and msg_id is not None:
        for prev_msg in history:
            if prev_msg.get("messageID") == msg_id and prev_msg.get("timestamp") == ts_now:
                replay_score = 1.0
                evidence.append(f"Replay check: Duplicate message ID {msg_id} and timestamp {ts_now} detected in history.")
                break
                
    # 5. Fallback to existing payload-based duplication check if not already flagged
    if replay_score < 1.0:
        for other_sender, other_history in history_store._history.items():
            for prev_msg in other_history:
                if (
                    prev_msg["x"] == msg["x"]
                    and prev_msg["y"] == msg["y"]
                    and prev_msg["speed"] == msg["speed"]
                    and prev_msg["heading"] == msg["heading"]
                ):
                    if prev_msg["timestamp"] == ts_now and other_sender == sender_id:
                        replay_score = 1.0
                        evidence.append("Replay check: Identical message duplicated/re-transmitted.")
                        break
                    else:
                        replay_score = max(replay_score, 0.9)
                        evidence.append(f"Replay check: Identical payload matches past report from sender '{other_sender}'.")
                        break
            if replay_score == 1.0:
                break

    sybil_score = 0.0
    for other_sender, other_history in history_store._history.items():
        if other_sender == sender_id:
            continue
        if len(other_history) == 0:
            continue
        prev_msg = other_history[-1]
        if abs(prev_msg["timestamp"] - ts_now) < 1000.0:
            dist = ((prev_msg["x"] - msg["x"]) ** 2 + (prev_msg["y"] - msg["y"]) ** 2) ** 0.5
            if dist < 2.0:
                sim = max(0.0, 1.0 - dist / 2.0)
                sybil_score = max(sybil_score, sim)
                evidence.append(f"Sybil check: Co-located vehicle identity '{other_sender}' reported position within {dist:.2f}m.")

    collusion_score = 0.0
    event = msg.get("event")
    if event:
        co_reporters = 0
        for other_sender, other_history in history_store._history.items():
            if other_sender == sender_id:
                continue
            for prev_msg in other_history:
                if abs(prev_msg["timestamp"] - ts_now) < 5000.0:
                    dist = ((prev_msg["x"] - msg["x"]) ** 2 + (prev_msg["y"] - msg["y"]) ** 2) ** 0.5
                    if dist < 100.0 and prev_msg.get("event") == event:
                        co_reporters += 1
                        break
        if co_reporters > 0:
            collusion_score = min(1.0, co_reporters * 0.25)
            evidence.append(f"Collusion check: {co_reporters} other sender(s) in range reporting the same event type '{event}'.")

    if not plausibility_pass:
        anomaly_score = 1.0
    else:
        kinematic_anomaly = 1.0 - kinematic_score
        temporal_anomaly = 1.0 - temporal_consistency
        anomaly_score = max(
            kinematic_anomaly * 0.6,
            temporal_anomaly * 0.5,
            replay_score * 0.8,
            sybil_score * 0.8,
            collusion_score * 0.4,
        )
        anomaly_score = min(1.0, max(0.0, anomaly_score))

    if len(history) == 0:
        behavior_evidence_quality = 0.2
    else:
        behavior_evidence_quality = round(min(1.0, 0.2 + 0.8 * (len(history) / MAX_HISTORY)), 2)

    cert_score = certificate_rotation_score(cert_rotation_anomaly)
    if cert_score is not None and cert_score > 0:
        evidence.append("Certificate rotation: anomalous rotation rate flagged by B1's VehicleStateManager (delegated).")

    passed = plausibility_pass and kinematics_ok and (replay_score < 0.9)

    result = MBDResult(
        passed=passed,
        behavior_evidence_quality=float(behavior_evidence_quality),
        kinematic_score=float(kinematic_score),
        temporal_consistency=float(temporal_consistency),
        replay_score=float(replay_score),
        sybil_score=float(sybil_score),
        collusion_score=float(collusion_score),
        anomaly_score=float(anomaly_score),
        evidence=evidence,
        certificate_rotation_score=cert_score,
        boundary="MBD",  # renamed from "B2_MBD" -- see module docstring
        sender=sender_id,
        speed_valid=plausibility["speed_valid"],
        timestamp_valid=plausibility["timestamp_valid"],
        position_valid=plausibility["position_valid"],
        heading_valid=plausibility["heading_valid"],
        kinematics_ok=kinematics_ok,
        kinematics_detail=kinematics_detail,
        mbd_pass=passed,
    )

    history_store.push(sender_id, msg)
    return result
