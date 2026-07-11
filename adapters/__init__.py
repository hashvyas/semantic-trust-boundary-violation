"""
adapters package
==================
Converts FinalTrustDecision into downstream-consumable formats. See
adapters/base.py for the contract every adapter must follow (zero trust
logic -- pure format conversion of an already-final decision).
"""

from adapters.base import Adapter
from adapters.logging_adapter import LoggingAdapter
from adapters.api_adapter import APIAdapter
from adapters.ds_mass_adapter import DSMassAdapter, DSMassOutput

__all__ = [
    "Adapter",
    "LoggingAdapter",
    "APIAdapter",
    "DSMassAdapter",
    "DSMassOutput",
]
