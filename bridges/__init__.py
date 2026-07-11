"""bridges package: format-translation adapters between this repo's
native message schemas and external module schemas. Currently contains
message_adapter.py (ETSI CAM -> flat MBD/CP schema). Leaf module -- must
never import from any trust-stack layer (b1_scsv, b2_explain, mbd, cp,
pki, trust_engine, adapters, b3/pipeline.b3_bridge)."""

from bridges.message_adapter import ProjectionOrigin, to_flat_report, project_to_local_meters

__all__ = ["ProjectionOrigin", "to_flat_report", "project_to_local_meters"]
