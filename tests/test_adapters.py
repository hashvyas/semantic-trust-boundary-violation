"""
tests/test_adapters.py
=========================
Standalone (no pytest) tests for the adapters layer: pure-format-
conversion contract, dependency isolation (adapters must only import
trust_engine.models, never b1/b2/b3), and numerical correctness of the
DS MASS mapping.

Run with: python3 tests/test_adapters.py
"""

from __future__ import annotations

import ast
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from adapters.logging_adapter import LoggingAdapter
from adapters.api_adapter import APIAdapter
from adapters.ds_mass_adapter import DSMassAdapter
from trust_engine.models import FinalTrustDecision, TrustLevel, SemanticRisk

_FAILURES = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        _FAILURES.append(name)


def decision(**overrides):
    base = dict(
        trust_score=0.8, trust_level=TrustLevel.ACCEPT,
        semantic_risk=SemanticRisk.NONE, cryptographic_risk="low",
        attack_detected=False, confidence=0.9, reasoning="test",
        contributors=["B1", "B2"], details={},
    )
    base.update(overrides)
    return FinalTrustDecision(**base)


# -- Dependency isolation: adapters must only import trust_engine.models,
#    never b1_scsv / b2_explain / pipeline.b3_bridge internals.
ROOT = pathlib.Path(__file__).resolve().parent.parent
FORBIDDEN_PREFIXES = ("b1_scsv", "b2_explain", "pipeline.b3_bridge", "b3.")
for f in (ROOT / "adapters").glob("*.py"):
    tree = ast.parse(f.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    violations = [m for m in imported if any(m.startswith(p) for p in FORBIDDEN_PREFIXES)]
    check(f"{f.name}: no forbidden cross-layer imports", not violations)


# -- LoggingAdapter --------------------------------------------------------
d = decision(trust_level=TrustLevel.REJECT, attack_detected=True)
log = LoggingAdapter().adapt(d)
check("LoggingAdapter output is a dict", isinstance(log, dict))
check("LoggingAdapter preserves trust_level", log["trust_level"] == "REJECT")
check("LoggingAdapter preserves attack_detected", log["attack_detected"] is True)
check("LoggingAdapter has timestamp", "timestamp" in log)

# -- APIAdapter --------------------------------------------------------------
api = APIAdapter().adapt(d)
check("APIAdapter output has api_version", api.get("api_version") == "1.0")
check("APIAdapter status matches trust_level", api["result"]["status"] == "REJECT")
check("APIAdapter does not leak internal field names",
      "trust_level" not in api["result"] and "trust_score" not in api["result"])

# -- DSMassAdapter -----------------------------------------------------------
for trust, conf in [(1.0, 1.0), (0.0, 1.0), (0.5, 0.5), (0.8, 0.9), (0.0, 0.0), (1.0, 0.0)]:
    d2 = decision(trust_score=trust, confidence=conf)
    out = DSMassAdapter().adapt(d2)
    total = out.m_A + out.m_not_A + out.m_Theta
    check(f"DSMassAdapter sums to 1.0 (trust={trust}, conf={conf})", abs(total - 1.0) < 1e-9)
    check(f"DSMassAdapter all masses non-negative (trust={trust}, conf={conf})",
          out.m_A >= 0 and out.m_not_A >= 0 and out.m_Theta >= 0)

# Full confidence + full trust -> all mass on benign, zero ignorance
out = DSMassAdapter().adapt(decision(trust_score=1.0, confidence=1.0))
check("DSMassAdapter: perfect trust+confidence -> m_A=1, m_Theta=0",
      abs(out.m_A - 1.0) < 1e-9 and abs(out.m_Theta) < 1e-9)

# Zero confidence -> all mass is ignorance, regardless of trust_score
out = DSMassAdapter().adapt(decision(trust_score=0.9, confidence=0.0))
check("DSMassAdapter: zero confidence -> m_Theta=1 regardless of trust_score",
      abs(out.m_Theta - 1.0) < 1e-9)

# -- Adapters must be pure functions: same input -> same output ------------
d3 = decision()
out1 = DSMassAdapter().adapt(d3)
out2 = DSMassAdapter().adapt(d3)
check("DSMassAdapter is deterministic/pure", out1.to_dict() == out2.to_dict())


print()
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S): {_FAILURES}")
    sys.exit(1)
print("All adapter checks passed.")
