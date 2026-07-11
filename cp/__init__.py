"""cp package: Cooperative Perception layer. Fifth stage of the frozen
V2X Trust Stack. See cp/cp_layer.py for the module-level contract.

NOTE: modules/cp.py (the uploaded earlier draft) was intentionally NOT
carried into this package -- see cp/cp_layer.py's module docstring and
responsibility-audit finding D4. cp_layer.py is the sole CP
implementation in this repository."""

from cp.cp_layer import cp_layer, spatial_consistency, speed_consistency, heading_consistency, source_diversity

__all__ = ["cp_layer", "spatial_consistency", "speed_consistency", "heading_consistency", "source_diversity"]
