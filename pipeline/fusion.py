"""
pipeline/fusion.py
==================
DEPRECATED. All fusion logic now lives exclusively in
trust_engine/decision_engine.py (TrustDecisionEngine.decide()) and
trust_engine/policy.py (TrustPolicy).

This module is kept as a thin, clearly-labeled compatibility shim ONLY
so that any external code still importing `fuse_results` directly does
not hard-crash. It contains NO independent trust logic of its own --
it delegates to TrustDecisionEngine and reshapes the typed
FinalTrustDecision back into the legacy (decision, reason, fusion_details)
tuple shape.

pipeline/orchestrator.py does NOT use this module -- it calls
TrustDecisionEngine directly. Do not add new logic here.
"""

from __future__ import annotations
import warnings
from typing import Any, Dict, Tuple

from trust_engine.decision_engine import TrustDecisionEngine

_ENGINE = TrustDecisionEngine()


def fuse_results(
    b1_result: Dict[str, Any],
    b2_result: Dict[str, Any],
    b3_result: Dict[str, Any]
) -> Tuple[str, str, Dict[str, Any]]:
    """DEPRECATED shim. Delegates to TrustDecisionEngine.decide().

    Use trust_engine.decision_engine.TrustDecisionEngine.decide() directly
    in new code -- it returns a typed FinalTrustDecision instead of this
    legacy tuple.
    """
    warnings.warn(
        "pipeline.fusion.fuse_results is deprecated; use "
        "trust_engine.decision_engine.TrustDecisionEngine.decide() instead. "
        "This shim delegates to it and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    decision = _ENGINE.decide(b1_result, b2_result, b3_result)
    d = decision.to_dict()
    return d["trust_level"], d["reasoning"], d
