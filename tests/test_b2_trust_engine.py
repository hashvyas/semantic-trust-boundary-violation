"""
tests/test_b2_trust_engine.py
================================
Standalone (no pytest dependency) regression tests for the new
B2 Explainability layer and Trust Decision Engine.

Run with: python3 tests/test_b2_trust_engine.py
Exits non-zero on any assertion failure.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from b2_explain.explainability import ExplainabilityEngine
from trust_engine.decision_engine import TrustDecisionEngine
from trust_engine.models import TrustLevel, SemanticRisk
from trust_engine.policy import TrustPolicy

_FAILURES = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        _FAILURES.append(name)


def va(**overrides):
    base = {"valid": True, "fatal": False, "score": 1.0, "confidence": 1.0,
            "reasons": [], "checks": {}, "details": {}}
    base.update(overrides)
    return base


def b3(**overrides):
    base = {"available": False, "label": None, "confidence": None, "status": "n/a"}
    base.update(overrides)
    return base


b2 = ExplainabilityEngine()
te = TrustDecisionEngine()


# -- B2: must not accept None, must not touch raw payload -----------------
try:
    b2.explain(None)
    check("B2 rejects None input", False)
except ValueError:
    check("B2 rejects None input", True)

report = b2.explain(va())
check("B2 report has no raw-payload leakage keys",
      set(report.to_dict().keys()) == {
          "explanation_text", "evidence", "confidence_calibration",
          "provenance", "validation_valid", "validation_score"})

report_checks = b2.explain(va(checks={"replay": False, "signature": True}))
check("B2 evidence count matches checks dict", len(report_checks.evidence) == 2)
check("B2 never overrides B1 verdict (valid passthrough)",
      report_checks.validation_valid is True)

report_reasons_only = b2.explain(va(valid=False, reasons=["stale timestamp"]))
check("B2 falls back to reasons when no checks present",
      len(report_reasons_only.evidence) == 1
      and report_reasons_only.evidence[0].factor == "stale timestamp")

report_empty = b2.explain(va())
check("B2 never produces zero evidence items", len(report_empty.evidence) >= 1)


# -- Trust Engine: Rule 1 — B1 fatal -> REJECT regardless of B3 -----------
d = te.decide(va(fatal=True, valid=False, score=0.0, reasons=["impossible_speed"]),
              report.to_dict(), b3(available=True, label="MALICIOUS", confidence=0.01))
check("Rule1: B1 fatal forces REJECT even if B3 says benign",
      d.trust_level == TrustLevel.REJECT)
check("Rule1: B3 not in contributors when B1 fatal",
      "B3" not in d.contributors or d.details.get("b3_label") is not None)

# -- Rule 2: B1 pass + B3 high confidence -> REJECT ------------------------
d = te.decide(va(score=1.0), report.to_dict(),
              b3(available=True, label="MALICIOUS", confidence=0.95))
check("Rule2: high-confidence semantic attack forces REJECT",
      d.trust_level == TrustLevel.REJECT)
check("Rule2: attack_detected flag set", d.attack_detected is True)

# -- Rule 3: B1 pass + B3 medium confidence -> CAUTION --------------------
d = te.decide(va(score=1.0), report.to_dict(),
              b3(available=True, label="MALICIOUS", confidence=0.70))
check("Rule3: medium-confidence semantic anomaly -> CAUTION",
      d.trust_level == TrustLevel.CAUTION)

# -- Rule 4: B1 pass + B3 low confidence -> mild CAUTION -------------------
d = te.decide(va(score=1.0), report.to_dict(),
              b3(available=True, label="MALICIOUS", confidence=0.30))
check("Rule4: low-confidence semantic signal -> CAUTION (not silently ACCEPT)",
      d.trust_level == TrustLevel.CAUTION)

# -- Rule 5: both benign -> ACCEPT -----------------------------------------
d = te.decide(va(score=1.0), report.to_dict(), b3(available=False))
check("Rule5: clean B1 + unavailable B3 -> ACCEPT", d.trust_level == TrustLevel.ACCEPT)

d = te.decide(va(score=1.0), report.to_dict(),
              b3(available=True, label="BENIGN", confidence=0.99))
check("Rule5b: clean B1 + benign B3 label -> ACCEPT", d.trust_level == TrustLevel.ACCEPT)

# -- Regression: non-fatal but low/borderline B1 score must NOT silently
#    default to ACCEPT just because B3 is unavailable/benign. This was a
#    real gap found during integration testing (b1_score=0.8 previously
#    always produced ACCEPT regardless of score banding).
d = te.decide(va(valid=False, score=0.30, fatal=False, reasons=["cert_rotation_anomaly"]),
              b2.explain(va(valid=False, score=0.30, fatal=False, reasons=["cert_rotation_anomaly"])).to_dict(),
              b3(available=False))
check("Regression: low non-fatal B1 score (0.30) -> REJECT via crypto band",
      d.trust_level == TrustLevel.REJECT)

d = te.decide(va(valid=False, score=0.55, fatal=False, reasons=["minor anomaly"]),
              b2.explain(va(valid=False, score=0.55, fatal=False, reasons=["minor anomaly"])).to_dict(),
              b3(available=False))
check("Regression: mid non-fatal B1 score (0.55) -> CAUTION via crypto band",
      d.trust_level == TrustLevel.CAUTION)

# -- Conservative combination: bad B1 score + benign B3 must still degrade
d = te.decide(va(valid=False, score=0.50, fatal=False),
              b2.explain(va(valid=False, score=0.50, fatal=False)).to_dict(),
              b3(available=True, label="BENIGN", confidence=0.99))
check("Combination: mid crypto score wins over benign B3 (most conservative)",
      d.trust_level == TrustLevel.CAUTION)

# -- Conservative combination: good B1 score + malicious B3 must still degrade
d = te.decide(va(score=1.0), report.to_dict(),
              b3(available=True, label="MALICIOUS", confidence=0.95))
check("Combination: high B3 risk wins over perfect crypto score (most conservative)",
      d.trust_level == TrustLevel.REJECT)

# -- to_dict() round-trips without raising on all TrustLevel/SemanticRisk values
for tl in TrustLevel:
    check(f"TrustLevel.{tl.name} has string value", isinstance(tl.value, str))
for sr in SemanticRisk:
    check(f"SemanticRisk.{sr.name} has string value", isinstance(sr.value, str))

d_dict = d.to_dict()
check("FinalTrustDecision.to_dict() is JSON-serializable-shaped",
      all(isinstance(k, str) for k in d_dict.keys()))

# -- Policy config loading ---------------------------------------------------
custom = TrustPolicy.from_config({"trust_engine": {"semantic_high_confidence": 0.99}})
check("TrustPolicy.from_config overrides threshold", custom.semantic_high_confidence == 0.99)
check("TrustPolicy.from_config keeps default for unset field",
      custom.semantic_medium_confidence == 0.60)

default_policy = TrustPolicy.from_config(None)
check("TrustPolicy.from_config(None) returns defaults",
      default_policy.semantic_high_confidence == 0.85)


print()
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S): {_FAILURES}")
    sys.exit(1)
print("All checks passed.")