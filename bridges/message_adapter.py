"""
bridges/message_adapter.py
=============================
Translates this repo's native CAM message shape (nested, ETSI-style
fixed-point lat/lon in 1e-7 degrees, e.g. lat=485512345 -> 48.5512345N)
into the flat {sender, x, y, speed, heading, timestamp, event} schema
that MBD (mbd/mbd_layer.py) and CP (cp/cp_layer.py) expect -- both were
uploaded already assuming a local Cartesian (x, y) frame in meters, not
geodetic coordinates.

THIS IS A REAL, LOAD-BEARING CORRECTNESS CONCERN, not a formality: MBD's
Sybil check uses `dist < 2.0` (meters) and CP's spatial_consistency
divides spread by 20 (meters). Feeding raw lat/lon fixed-point integers
into those formulas directly would silently produce nonsense (values
differing by hundreds of thousands, not meters) -- every MBD/CP score
would be wrong in a way that wouldn't obviously crash, just silently
mis-detect everything. This adapter exists specifically to prevent that.

Projection used: equirectangular approximation (flat-Earth, referenced
to a single origin point) via:
    x = (lon - lon0) * cos(lat0) * EARTH_RADIUS_M
    y = (lat - lat0) * EARTH_RADIUS_M
This is accurate to well under 1% error for the ranges MBD/CP actually
reason about (tens to low hundreds of meters between nearby vehicles),
which is what their fixed thresholds (dist < 2.0, dist < 100.0, spread/20)
are calibrated for. It is NOT accurate over city-scale or longer
distances -- if this repo's scenarios ever span kilometers, a proper
geodesic/UTM projection should replace this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

_EARTH_RADIUS_M = 6_371_000.0
_ETSI_LATLON_SCALE = 1e-7  # ETSI CAM fixed-point degrees scale


@dataclass(frozen=True)
class ProjectionOrigin:
    """Reference point (radians) all x/y coordinates are relative to."""

    lat_rad: float
    lon_rad: float

    @staticmethod
    def from_degrees(lat_deg: float, lon_deg: float) -> "ProjectionOrigin":
        return ProjectionOrigin(math.radians(lat_deg), math.radians(lon_deg))


def _etsi_fixed_point_to_degrees(value: int) -> float:
    """Converts an ETSI CAM fixed-point lat/lon integer to decimal degrees."""
    return value * _ETSI_LATLON_SCALE


def project_to_local_meters(
    lat_deg: float, lon_deg: float, origin: ProjectionOrigin
) -> tuple:
    """Equirectangular projection to local (x, y) meters relative to origin."""
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    x = (lon_rad - origin.lon_rad) * math.cos(origin.lat_rad) * _EARTH_RADIUS_M
    y = (lat_rad - origin.lat_rad) * _EARTH_RADIUS_M
    return x, y


def _extract_field(msg: Dict[str, Any], *candidates: str, default: Any = None) -> Any:
    """Tries several possible key paths (this repo's CAM messages have
    varied slightly across test fixtures -- some nest under 'cam', some
    don't). Returns the first match."""
    for path in candidates:
        node: Any = msg
        ok = True
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if ok and node is not None:
            return node
    return default


def to_flat_report(
    cam_message: Dict[str, Any],
    origin: ProjectionOrigin,
    event: Optional[str] = None,
) -> Dict[str, Any]:
    """Converts one native CAM message dict into MBD/CP's flat schema.

    Raises ValueError (does not silently substitute defaults) if a
    required field is missing -- a silently-defaulted position/speed
    would corrupt every MBD/CP score derived from it, which is worse
    than failing loudly here.
    """
    if "x" in cam_message and "y" in cam_message:
        speed_val = float(cam_message["speed"])
        if cam_message.get("source") == "veremi":
            speed_val *= 3.6
        return {
            "sender": cam_message.get("sender"),
            "x": float(cam_message["x"]),
            "y": float(cam_message["y"]),
            "speed": speed_val,
            "heading": float(cam_message["heading"]) % 360.0,
            "timestamp": float(cam_message["timestamp"]),
            "event": event or cam_message.get("event"),
            "source": cam_message.get("source"),
        }

    sender = _extract_field(cam_message, "header.station_id", "cam.station_id", "station_id")
    lat_raw = _extract_field(
        cam_message,
        "cam.cam_parameters.basic_container.reference_position.latitude",
        "cam.latitude", "latitude", "lat",
    )
    lon_raw = _extract_field(
        cam_message,
        "cam.cam_parameters.basic_container.reference_position.longitude",
        "cam.longitude", "longitude", "lon",
    )
    speed_raw = _extract_field(
        cam_message,
        "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.speed",
        "cam.speed", "speed",
    )
    heading_raw = _extract_field(
        cam_message,
        "cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.heading",
        "cam.heading", "heading",
    )
    timestamp = _extract_field(
        cam_message, "cam.generation_delta_time", "timestamp", "generation_delta_time"
    )

    missing = [
        name
        for name, val in [
            ("sender/station_id", sender), ("latitude", lat_raw), ("longitude", lon_raw),
            ("speed", speed_raw), ("heading", heading_raw), ("timestamp", timestamp),
        ]
        if val is None
    ]
    if missing:
        raise ValueError(
            f"to_flat_report: message missing required field(s) {missing}; "
            f"cannot safely convert to MBD/CP schema (refusing to silently "
            f"default position/kinematic fields)."
        )

    # ETSI fixed-point CAM lat/lon are large integers (e.g. 485512345);
    # already-decimal degrees (e.g. 48.55) are passed through unchanged.
    lat_deg = _etsi_fixed_point_to_degrees(lat_raw) if abs(lat_raw) > 1000 else float(lat_raw)
    lon_deg = _etsi_fixed_point_to_degrees(lon_raw) if abs(lon_raw) > 1000 else float(lon_raw)
    x, y = project_to_local_meters(lat_deg, lon_deg, origin)

    # Speed: ETSI CAM speed is in 0.01 m/s units; convert to km/h to match
    # MBD/CP's MAX_SPEED_KMH=180 convention. Values already in a plausible
    # km/h range (<= 300) are passed through unchanged.
    speed_kmh = (speed_raw * 0.01 * 3.6) if abs(speed_raw) > 300 else float(speed_raw)

    # Heading: ETSI CAM heading is in 0.1 degree units (0-3600); normalize
    # to 0-360. Values already in range are passed through unchanged.
    heading_deg = (heading_raw / 10.0) if abs(heading_raw) > 360 else float(heading_raw)

    return {
        "sender": sender,
        "x": x,
        "y": y,
        "speed": speed_kmh,
        "heading": heading_deg % 360.0,
        "timestamp": float(timestamp),
        "event": event,
        "source": cam_message.get("source"),
    }
