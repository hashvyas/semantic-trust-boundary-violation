"""
b2_explain/config.py
=======================
Configuration for the B2 Explainability layer. Externalizes the
check-name -> human-readable-description table and calibration
defaults so they're editable without touching explainability.py logic,
and optionally overridable from isce_config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

#: Default human-readable descriptions for known B1 check/reason keys.
#: Extend as B1 grows new check types -- never branch on raw payload
#: content here, only on check/reason *names*.
DEFAULT_CHECK_DESCRIPTIONS: Dict[str, str] = {
    "replay": "Message replay/duplicate-detection check.",
    "stale_timestamp": "Timestamp freshness check against the validity window.",
    "impossible_kinematics": "Kinematic plausibility check (speed/acceleration/heading).",
    "cert_rotation_anomaly": "Certificate rotation-rate anomaly check.",
    "signature": "Cryptographic signature verification.",
    "revocation": "Certificate revocation status check.",
    "pki_signature": "Cryptographic signature verification.",
    "pki_revocation": "Certificate revocation status check.",
    "structural": "Message structural/schema validation.",
    "protocol_compliance": "Protocol compliance check against the message standard.",
}


@dataclass(frozen=True)
class B2Config:
    """Configuration for ExplainabilityEngine.

    check_descriptions:
        Mapping of B1 check name -> human-readable description. Merged
        on top of DEFAULT_CHECK_DESCRIPTIONS (config entries win).
    sparse_evidence_confidence_cap:
        Confidence ceiling applied when no evidence factors were found
        at all (i.e. B2 has very little to explain from).
    """

    check_descriptions: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CHECK_DESCRIPTIONS)
    )
    sparse_evidence_confidence_cap: float = 0.5

    @staticmethod
    def from_config(config: Optional[Dict[str, Any]]) -> "B2Config":
        if not config:
            return B2Config()
        b2_cfg = config.get("b2_explain", {}) if isinstance(config, dict) else {}
        merged = dict(DEFAULT_CHECK_DESCRIPTIONS)
        merged.update(b2_cfg.get("check_descriptions", {}))
        return B2Config(
            check_descriptions=merged,
            sparse_evidence_confidence_cap=b2_cfg.get(
                "sparse_evidence_confidence_cap", 0.5
            ),
        )
