"""
tests/verify_dependency_graph.py
===================================
Mechanically verifies the architecture's dependency constraints by
AST-parsing every module in each layer, instead of relying on manual
grep during an audit. Run with: python3 tests/verify_dependency_graph.py
"""

from __future__ import annotations

import ast
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_FAILURES = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        _FAILURES.append(name)


def imports_of(pyfile: pathlib.Path):
    tree = ast.parse(pyfile.read_text())
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    return mods


def layer_imports(layer_dir: str, forbidden_prefixes):
    violations = {}
    for f in (ROOT / layer_dir).glob("*.py"):
        mods = imports_of(f)
        bad = [m for m in mods if any(m.startswith(p) for p in forbidden_prefixes)]
        if bad:
            violations[str(f.relative_to(ROOT))] = bad
    return violations


# B1 must not import B2, B3, trust_engine, or adapters.
v = layer_imports("b1_scsv", ("b2_explain", "pipeline.b3_bridge", "b3.", "trust_engine", "adapters", "mbd", "cp"))
check("B1 (b1_scsv/) imports nothing from B2/B3/trust_engine/adapters/MBD/CP", not v)

# B2 must not import B3, trust_engine, or adapters (and must not import raw payload sources).
# May import contracts (neutral shared TrustEvidence contract).
v = layer_imports("b2_explain", ("pipeline.b3_bridge", "b3.", "trust_engine", "adapters", "mbd", "cp", "pki"))
check("B2 (b2_explain/) imports nothing from B3/trust_engine/adapters/MBD/CP/PKI", not v)

# trust_engine must not import b1_scsv, b2_explain internals, or pipeline.b3_bridge,
# or adapters (it depends only on the dict/dataclass contracts, not on how
# they were produced). May import contracts.
v = layer_imports("trust_engine", ("b1_scsv", "b2_explain", "pipeline.b3_bridge", "b3.", "adapters", "mbd", "cp", "pki"))
check("trust_engine/ imports nothing from B1/B2/B3/adapters/MBD/CP/PKI", not v)

# adapters must depend only on trust_engine.models.
v = layer_imports("adapters", ("b1_scsv", "b2_explain", "pipeline.b3_bridge", "b3.", "pipeline.orchestrator", "mbd", "cp", "pki"))
check("adapters/ imports nothing from B1/B2/B3/orchestrator/MBD/CP/PKI", not v)

# PKI must not import anything from downstream layers.
v = layer_imports("pki", ("b1_scsv", "b2_explain", "mbd", "cp", "trust_engine", "adapters", "b3.", "pipeline.b3_bridge"))
check("PKI (pki/) imports nothing from downstream layers", not v)

# MBD must not import B2/CP/B3/trust_engine/adapters (may reference b1_scsv only
# via the orchestrator-level delegation, never a direct import).
v = layer_imports("mbd", ("b2_explain", "cp", "trust_engine", "adapters", "b3.", "pipeline.b3_bridge", "b1_scsv"))
check("MBD (mbd/) imports nothing from B1/B2/CP/B3/trust_engine/adapters", not v)

# CP must not import B1/MBD/B2/B3/trust_engine/adapters directly (only
# receives already-computed observation_weights via the orchestrator).
v = layer_imports("cp", ("b1_scsv", "mbd", "b2_explain", "trust_engine", "adapters", "b3.", "pipeline.b3_bridge"))
check("CP (cp/) imports nothing from B1/MBD/B2/B3/trust_engine/adapters", not v)

# contracts/ and bridges/ must be leaf modules -- zero layer imports.
v = layer_imports("contracts", ("b1_scsv", "b2_explain", "mbd", "cp", "pki", "trust_engine", "adapters", "b3.", "pipeline.b3_bridge"))
check("contracts/ is a leaf module (zero layer imports)", not v)
v = layer_imports("bridges", ("b1_scsv", "b2_explain", "mbd", "cp", "pki", "trust_engine", "adapters", "b3.", "pipeline.b3_bridge"))
check("bridges/ is a leaf module (zero layer imports)", not v)

# simulation/ (the renamed dispatcher, D5) must not import PKI/B1/MBD/CP
# internals directly -- it only reads FinalTrustDecision, per audit D5's
# resolution of the duplicated pki_pass/mbd_pass re-check.
v = layer_imports("simulation", ("b1_scsv", "b2_explain", "mbd", "cp", "pki", "pipeline.b3_bridge", "b3."))
check("simulation/ (dispatcher) imports nothing from PKI/B1/MBD/CP/B3 internals", not v)

for f in (ROOT / "adapters").glob("*.py"):
    if f.name in ("__init__.py",):
        continue
    mods = imports_of(f)
    te_mods = [m for m in mods if m.startswith("trust_engine")]
    check(f"adapters/{f.name} only touches trust_engine.models (not decision_engine/policy)",
          all(m == "trust_engine.models" or not m.startswith("trust_engine.decision_engine")
              and not m.startswith("trust_engine.policy") for m in te_mods)
          if te_mods else True)

# No circular imports: actually import every module and confirm no ImportError.
import importlib
MODULES_TO_IMPORT = [
    "b1_scsv.scsv", "b1_scsv.models",
    "b2_explain.explainability", "b2_explain.models", "b2_explain.evidence", "b2_explain.config",
    "trust_engine.decision_engine", "trust_engine.models", "trust_engine.policy", "trust_engine.exceptions",
    "pipeline.b3_bridge", "pipeline.synthesizer", "pipeline.orchestrator", "pipeline.fusion",
    "adapters", "adapters.base", "adapters.logging_adapter", "adapters.api_adapter", "adapters.ds_mass_adapter",
    "pki", "pki.pki_layer",
    "mbd", "mbd.mbd_layer",
    "cp", "cp.cp_layer",
    "bridges.message_adapter",
    "contracts", "contracts.trust_evidence",
    "simulation", "simulation.llm_dispatcher",
]
for mod in MODULES_TO_IMPORT:
    try:
        importlib.import_module(mod)
        check(f"import {mod} succeeds (no circular import)", True)
    except Exception as e:
        check(f"import {mod} succeeds (no circular import) -- {e}", False)


print()
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S): {_FAILURES}")
    sys.exit(1)
print("All dependency graph checks passed.")
