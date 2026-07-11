"""
tests/test_pki_mbd_cp_integration.py
=======================================
Integration tests for the PKI -> B1 -> MBD -> B2 -> CP -> B3 wiring,
and mechanical verification of the responsibility-audit duplication
resolutions (D1, D2, D3, D4, D5).

Run with: python3 tests/test_pki_mbd_cp_integration.py
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent

_FAILURES = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        _FAILURES.append(name)


# ============================================================
# Backward compatibility: default-mode pipeline is UNCHANGED
# ============================================================
from pipeline.orchestrator import ISCEPipeline

pipe_default = ISCEPipeline()
msg = json.load(open(ROOT / "test_messages/benign/normal_car.json"))
res = pipe_default.run([msg])
check("Default pipeline: pki key is None (PKI not silently faked)", res["pki"] is None)
check("Default pipeline: mbd key is None (MBD disabled by default)", res["mbd"] is None)
check("Default pipeline: cp key is None (CP disabled by default)", res["cp"] is None)
check("Default pipeline: decision still ACCEPT for benign fixture", res["decision"] == "ACCEPT")


# ============================================================
# D4: cp.py must not exist anywhere in this repository
# ============================================================
cp_py_files = list(ROOT.glob("**/cp.py"))
check("D4: no stray cp.py exists anywhere in the repo (only cp/cp_layer.py)",
      len(cp_py_files) == 0)


# ============================================================
# §0 label collision resolution: no "B1_PKI"/"B2_MBD"/"B3_CP" strings
# remain in the integrated pki/mbd/cp packages.
# ============================================================
for pkg, forbidden in [("pki", "B1_PKI"), ("mbd", "B2_MBD"), ("cp", "B3_CP")]:
    hits = []
    for f in (ROOT / pkg).glob("*.py"):
        text = f.read_text()
        # Only flag it as a live string literal, not a docstring reference
        # explaining the rename (those intentionally mention the old name).
        if f'"{forbidden}"' in text or f"'{forbidden}'" in text:
            # allow it if it's clearly inside a comment/docstring context
            # mentioning the rename -- heuristic: the line also contains
            # "rename" or "->" or "collision" nearby.
            for line in text.splitlines():
                if forbidden in line and not any(
                    kw in line for kw in ("renamed", "->", "collision", "resolves", "was")
                ):
                    hits.append((f.name, line.strip()))
    check(f"§0: no live '{forbidden}' string literal remains in {pkg}/", not hits)


# ============================================================
# D3: B1 must never import/reference MBD's VehicleHistoryStore
# (mechanically enforced, not just documented)
# ============================================================
b1_source = (ROOT / "b1_scsv" / "scsv.py").read_text()
tree = ast.parse(b1_source)
imported_names = []
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module:
        imported_names.append(node.module)
    elif isinstance(node, ast.Import):
        imported_names.extend(a.name for a in node.names)
check("D3: b1_scsv/scsv.py does not import mbd (single-message-only enforced)",
      not any(m == "mbd" or m.startswith("mbd.") for m in imported_names))
check("D3: b1_scsv/scsv.py source never references VehicleHistoryStore",
      "VehicleHistoryStore" not in b1_source)


# ============================================================
# D1: cert_rotation_owner flag -- default preserves exact old behavior,
# "mbd" mode still runs the check but doesn't penalize B1's score.
# ============================================================
from b1_scsv.scsv import SCSV

scsv_default = SCSV()
check("D1: SCSV default cert_rotation_owner is 'b1' (zero behavior change)",
      scsv_default._cert_rotation_owner == "b1")

scsv_mbd = SCSV(cert_rotation_owner="mbd")
check("D1: SCSV accepts cert_rotation_owner='mbd'", scsv_mbd._cert_rotation_owner == "mbd")

try:
    SCSV(cert_rotation_owner="invalid")
    check("D1: SCSV rejects invalid cert_rotation_owner", False)
except ValueError:
    check("D1: SCSV rejects invalid cert_rotation_owner", True)

check("D1: SCSV exposes check_cert_rotation_for_station (MBD delegation accessor)",
      hasattr(scsv_mbd, "check_cert_rotation_for_station"))
check("D1: accessor returns False for unknown station (no false positive)",
      scsv_mbd.check_cert_rotation_for_station(999999) is False)


# ============================================================
# D2: both B1's structural replay check AND MBD's behavioral replay_score
# fire independently on the same replay scenario (complementary, not
# merged into one -- see audit D2 resolution).
# ============================================================
from mbd import mbd_layer, VehicleHistoryStore
from bridges.message_adapter import to_flat_report, ProjectionOrigin

scsv_for_replay = SCSV()
history = VehicleHistoryStore()
origin = ProjectionOrigin.from_degrees(48.5512345, 9.6123456)

replay_msgs = json.load(open(ROOT / "test_messages/b1_fail/replay.json"))
b1_flagged_replay = False
mbd_flagged_replay = False
for m in replay_msgs:
    b1_res = scsv_for_replay.check_stateful(m)
    if "Replay detected" in (b1_res.reasons or []) or not b1_res.checks.get("replay", True):
        b1_flagged_replay = True
    flat = to_flat_report(m, origin)
    mbd_res = mbd_layer(flat, history)
    if mbd_res["replay_score"] > 0.0:
        mbd_flagged_replay = True

check("D2: B1's structural replay cache flags the replay scenario", b1_flagged_replay)
check("D2: MBD's behavioral replay_score also flags the replay scenario", mbd_flagged_replay)


# ============================================================
# PKI unit tests
# ============================================================
from pki import CertificateAuthority, pki_layer, sign_message

ca = CertificateAuthority()
priv, pub, cert = ca.issue_certificate("veh_test")
sig = sign_message({"payload": "test"}, priv)
result = pki_layer({"payload": "test"}, sig, cert, pub, ca)
check("PKI: valid signature + valid cert -> pki_pass True", result["pki_pass"] is True)
check("PKI: boundary label is 'PKI' (not 'B1_PKI')", result["boundary"] == "PKI")

ca.mark_compromised("veh_test")
result2 = pki_layer({"payload": "test"}, sig, cert, pub, ca)
check("PKI: compromised-but-not-revoked cert still passes crypto checks "
      "(intentional PKI blind spot, per STBV threat model)",
      result2["pki_pass"] is True and result2["compromised"] is True)

ca.revoke("veh_test")
result3 = pki_layer({"payload": "test"}, sig, cert, pub, ca)
check("PKI: revoked cert fails pki_pass", result3["pki_pass"] is False)

bad_sig = sign_message({"payload": "tampered"}, priv)
ca2 = CertificateAuthority()
_, pub2, cert2 = ca2.issue_certificate("veh_test2")
result4 = pki_layer({"payload": "test"}, bad_sig, cert2, pub2, ca2)
check("PKI: mismatched signature fails sig_valid", result4["sig_valid"] is False)


# ============================================================
# MBD unit tests
# ============================================================
from mbd import MBDResult

fresh_history = VehicleHistoryStore()
report1 = {"sender": 1, "x": 0.0, "y": 0.0, "speed": 50.0, "heading": 90.0, "timestamp": 0.0, "event": None}
r1 = mbd_layer(report1, fresh_history)
check("MBD: first message for a sender has no history -> passed", r1["passed"] is True)
check("MBD: boundary label is 'MBD' (not 'B2_MBD')", r1["boundary"] == "MBD")

# Implausible kinematic jump: same sender, huge speed jump
report2 = {"sender": 1, "x": 500.0, "y": 500.0, "speed": 180.0, "heading": 300.0, "timestamp": 0.1, "event": None}
r2 = mbd_layer(report2, fresh_history)
check("MBD: implausible kinematic jump -> low kinematic_score", r2["kinematic_score"] < 0.5)


# ============================================================
# CP unit tests
# ============================================================
from cp import cp_layer

reports = [
    {"sender": 1, "x": 0.0, "y": 0.0, "speed": 50.0, "heading": 90.0},
    {"sender": 2, "x": 1.0, "y": 1.0, "speed": 51.0, "heading": 91.0},
    {"sender": 3, "x": 0.5, "y": 0.5, "speed": 50.5, "heading": 90.5},
]
cp_res = cp_layer(reports)
check("CP: boundary label is 'CP' (not 'B3_CP')", cp_res["boundary"] == "CP")
check("CP: consistent reports -> cp_pass True", cp_res["cp_pass"] is True)

cp_empty = cp_layer([])
check("CP: empty report list handled gracefully", cp_empty["num_reports"] == 0)

# Weighted fusion: a low-weight outlier should influence the fused score less
outlier_reports = reports + [{"sender": 4, "x": 500.0, "y": 500.0, "speed": 5.0, "heading": 10.0}]
unweighted = cp_layer(outlier_reports)
weighted = cp_layer(outlier_reports, observation_weights={4: 0.01})
check("CP: down-weighting an outlier improves spatial consistency vs unweighted",
      weighted["spatial_score"] >= unweighted["spatial_score"])


# ============================================================
# contracts.TrustEvidence + B2.explain_evidence
# ============================================================
from contracts.trust_evidence import TrustEvidence
from b2_explain.explainability import ExplainabilityEngine

ev1 = TrustEvidence.from_validation_assessment({"valid": True, "score": 1.0, "confidence": 0.9, "reasons": []})
ev2 = TrustEvidence.from_mbd_result(dict(r1))
b2 = ExplainabilityEngine()
report = b2.explain_evidence([ev1, ev2])
check("B2.explain_evidence: combines multiple TrustEvidence sources",
      report.to_dict()["provenance"]["layer_count"] == 2)
check("B2.explain_evidence: never modifies upstream verdicts (validation_valid passthrough)",
      report.validation_valid == (ev1.passed and ev2.passed))

try:
    b2.explain_evidence([])
    check("B2.explain_evidence: rejects empty evidence list", False)
except ValueError:
    check("B2.explain_evidence: rejects empty evidence list", True)


# ============================================================
# CP integration into Trust Decision Engine (A1 Regression)
# ============================================================
from trust_engine.decision_engine import TrustDecisionEngine
from trust_engine.models import TrustLevel

# CP must change the final decision when its evidence is poor
cp_bad = {"cp_pass": False, "fusion_confidence": 0.2, "cp_confidence": 0.2,
          "num_reports": 5, "senders": [1,2,3]}
cp_good = {"cp_pass": True, "fusion_confidence": 0.95, "cp_confidence": 0.95,
           "num_reports": 5, "senders": [1,2,3]}
b1_clean = {"valid": True, "fatal": False, "score": 1.0, "confidence": 1.0, "reasons": [], "checks": {}, "details": {}}
b2_engine = ExplainabilityEngine()
b2_base = b2_engine.explain(b1_clean).to_dict()

def fold_cp(b2_dict, cp_dict):
    ev = TrustEvidence.from_cp_result(cp_dict)
    return {
        **b2_dict, 
        "validation_score": (b2_dict["validation_score"] + ev.score) / 2.0,
        "validation_valid": b2_dict["validation_valid"] and ev.passed,
        "provenance": {
            **b2_dict.get("provenance", {}),
            "source_layers": b2_dict.get("provenance", {}).get("source_layers", []) + ["CP"]
        }
    }

te = TrustDecisionEngine()
d_bad_cp = te.decide(b1_clean, fold_cp(b2_base, cp_bad), {"available": False, "risk_level": "unavailable"})
d_good_cp = te.decide(b1_clean, fold_cp(b2_base, cp_good), {"available": False, "risk_level": "unavailable"})

check("CP: poor fusion confidence degrades trust_level vs good fusion",
      d_bad_cp.trust_level != TrustLevel.ACCEPT or d_bad_cp.trust_score < d_good_cp.trust_score)
check("CP present appears in contributors", "CP" in d_bad_cp.contributors)
# ============================================================
# CP Regression Test (A1) -- Mathematical Proof of Fusion
# ============================================================
from contracts.trust_evidence import TrustEvidence as _TE
from trust_engine.models import TrustLevel as _TL
from trust_engine.decision_engine import TrustDecisionEngine

def _fold_cp(b2_dict, cp_dict):
    ev = _TE.from_cp_result(cp_dict)
    return {**b2_dict, "validation_score": (b2_dict["validation_score"] + ev.score) / 2.0,
            "validation_valid": b2_dict["validation_valid"] and ev.passed,
            "provenance": {**b2_dict["provenance"], "source_layers":
                list(b2_dict["provenance"].get("source_layers", [])) + ["CP"]}}

cp_bad = {"cp_pass": False, "fusion_confidence": 0.2, "cp_confidence": 0.2, "num_reports": 5, "senders": [1,2,3]}
cp_good = {"cp_pass": True, "fusion_confidence": 0.95, "cp_confidence": 0.95, "num_reports": 5, "senders": [1,2,3]}
_b1c = {"valid": True, "fatal": False, "score": 1.0, "confidence": 1.0, "reasons": [], "checks": {}, "details": {}}
_b2b = ExplainabilityEngine().explain(_b1c).to_dict()
_te = TrustDecisionEngine()

_d_bad = _te.decide(_b1c, _fold_cp(_b2b, cp_bad), {"available": False, "risk_level": "unavailable"})
_d_good = _te.decide(_b1c, _fold_cp(_b2b, cp_good), {"available": False, "risk_level": "unavailable"})

check("A1: poor CP fusion confidence degrades trust vs good CP",
       _d_bad.trust_level != _TL.ACCEPT or _d_bad.trust_score < _d_good.trust_score)
check("A1: CP appears in contributors when folded in", "CP" in _d_bad.contributors)

print()
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S): {_FAILURES}")
    sys.exit(1)
print("All PKI/MBD/CP integration checks passed.")