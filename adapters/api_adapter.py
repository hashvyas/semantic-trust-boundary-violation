"""
adapters/api_adapter.py
=========================
Converts a FinalTrustDecision into a stable, versioned JSON-API-style
response envelope for external HTTP/RPC consumers. Pure format
conversion -- no trust logic. The envelope's shape is intentionally
decoupled from FinalTrustDecision's internal field names so that
internal refactors to trust_engine.models don't automatically break
external API consumers.
"""

from __future__ import annotations

from typing import Any, Dict

from adapters.base import Adapter
from trust_engine.models import FinalTrustDecision

_API_VERSION = "1.0"


class APIAdapter(Adapter):
    """Formats a FinalTrustDecision as a versioned API response body."""

    def adapt(self, decision: FinalTrustDecision) -> Dict[str, Any]:
        return {
            "api_version": _API_VERSION,
            "result": {
                "status": decision.trust_level.value,
                "score": round(decision.trust_score, 4),
                "risk": {
                    "semantic": decision.semantic_risk.value,
                    "cryptographic": decision.cryptographic_risk,
                },
                "attack_detected": decision.attack_detected,
                "confidence": round(decision.confidence, 4),
                "explanation": decision.reasoning,
                "evaluated_by": decision.contributors,
            },
        }
