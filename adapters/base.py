"""
adapters/base.py
==================
Base contract for all Adapters.

Adapters convert a FinalTrustDecision into a downstream-consumable
format. They contain ZERO trust logic -- no thresholds, no fusion, no
decisions about what counts as risky. If an adapter needs to branch on
trust_level/semantic_risk/etc., it may only do so to select an output
*format* (e.g. which field name to use), never to change the *meaning*
of the decision. All trust reasoning has already happened upstream in
trust_engine.decision_engine.TrustDecisionEngine.

Any new adapter must:
  1. Subclass Adapter.
  2. Implement adapt(decision) -> Any, a pure function of the
     FinalTrustDecision it's given (plus, optionally, static config
     supplied at construction time -- never additional trust computation).
  3. Not import anything from b1_scsv, b2_explain, or pipeline.b3_bridge.
     Adapters depend ONLY on trust_engine.models.FinalTrustDecision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trust_engine.models import FinalTrustDecision


class Adapter(ABC):
    """Abstract base class for all downstream-format adapters."""

    @abstractmethod
    def adapt(self, decision: FinalTrustDecision) -> Any:
        """Convert a FinalTrustDecision into this adapter's output format.

        Must be a pure function of `decision` (and any static config set
        at __init__ time). Must not consult B1/B2/B3 internals, and must
        not alter trust_level/trust_score/semantic_risk/etc. -- those are
        final by the time they reach an adapter.
        """
        raise NotImplementedError
