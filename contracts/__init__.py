"""contracts package: neutral, zero-dependency shared data contracts
that more than one layer needs to agree on (e.g. TrustEvidence, read by
both B2 and the Trust Decision Engine). Nothing in this package may
import from b1_scsv, b2_explain, mbd, cp, b3/pipeline.b3_bridge,
trust_engine, or adapters -- it is a leaf module by design."""

from contracts.trust_evidence import TrustEvidence

__all__ = ["TrustEvidence"]
