"""
trust_engine/exceptions.py
============================
Exception hierarchy for the Trust Decision Engine and its inputs.
"""

from __future__ import annotations


class TrustEngineError(Exception):
    """Base class for all trust_engine exceptions."""


class MissingLayerInputError(TrustEngineError):
    """Raised when TrustDecisionEngine.decide() is called without one of
    the three required inputs (ValidationAssessment, ExplainabilityReport,
    SemanticResult dicts)."""


class InvalidPolicyConfigError(TrustEngineError):
    """Raised when a TrustPolicy is constructed with out-of-range or
    inconsistent thresholds (e.g. reject_below >= caution_below)."""
