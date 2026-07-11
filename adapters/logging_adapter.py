"""
adapters/logging_adapter.py
=============================
Converts a FinalTrustDecision into a structured log record (dict, ready
for json.dumps or a logging handler's `extra=` parameter). Pure format
conversion -- no trust logic.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from adapters.base import Adapter
from trust_engine.models import FinalTrustDecision


class LoggingAdapter(Adapter):
    """Formats a FinalTrustDecision as a structured log record.

    Suitable for feeding into Python's `logging` module (as `extra=`),
    a JSON log shipper, or the harness's --log file output.
    """

    def __init__(self, source: str = "isce_pipeline") -> None:
        self.source = source

    def adapt(self, decision: FinalTrustDecision) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "source": self.source,
            "event": "trust_decision",
            "trust_level": decision.trust_level.value,
            "trust_score": decision.trust_score,
            "semantic_risk": decision.semantic_risk.value,
            "cryptographic_risk": decision.cryptographic_risk,
            "attack_detected": decision.attack_detected,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "contributors": list(decision.contributors),
        }
