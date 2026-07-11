"""simulation package: downstream vehicle/application decision layer.
Last stage of the frozen V2X Trust Stack. See
simulation/llm_dispatcher.py for the module-level contract.

Renamed from the uploaded modules/decision_engine.py per responsibility-
audit finding D5 -- "decision_engine" is reserved exclusively for
trust_engine/decision_engine.py."""

from simulation.llm_dispatcher import llm_dispatcher, compute_ttc

__all__ = ["llm_dispatcher", "compute_ttc"]
