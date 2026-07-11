# ISCE / STBV Pipeline — Team Onboarding Guide

**Purpose of this document:** get a new teammate from zero to productive
on this codebase — what it does, why it's built this way, how to run
it, how to extend it safely, and what's still unfinished. Read this
before touching any code.

---

## 1. What this project is

**STBV = Semantic Trust Boundary Violation.**

Traditional V2X (vehicle-to-everything) security — PKI, certificates,
signatures — can tell you *who* sent a message and whether it was
tampered with in transit. It cannot tell you whether the *content* of
the message is trustworthy. A perfectly-signed, perfectly-authenticated
message can still describe a fabricated emergency, a false hazard, or a
coordinated multi-vehicle lie.

The core research claim of this project: **trust should be able to
degrade at the boundary between layers**, specifically at the boundary
between cryptographic validation and semantic/AI-agent reasoning. A
message can pass PKI validation and still get downgraded to CAUTION or
REJECT if semantic analysis flags it as manipulative — that's the
system's whole reason for existing.

The system is called **ISCE** (Intent-Semantic-Consistency-Engine) in
the codebase.

## 2. The architecture, in one picture

```
Incoming V2X Message (CAM/DENM/CPM)
        │
        ▼
┌─────────────────────┐
│ B1 — Validation       │  Cryptographic / structural checks only:
│ (b1_scsv/)            │  cert, signature, revocation, replay,
│                        │  kinematic plausibility, timestamp freshness.
└─────────────────────┘  NEVER does semantic reasoning.
        │  ValidationAssessment
        ▼
┌─────────────────────┐
│ B2 — Explainability    │  Explains WHY B1 decided what it decided.
│ (b2_explain/)          │  Takes ONLY the ValidationAssessment —
│                        │  never sees the raw message.
└─────────────────────┘  NEVER changes B1's verdict. NEVER does
        │  ExplainabilityReport  semantic reasoning.
        ▼
┌─────────────────────┐
│ Synthesizer            │  Turns the raw message cluster into a
│ (pipeline/             │  deterministic natural-language scene
│  synthesizer.py)       │  description for B3 to read.
└─────────────────────┘
        │  scene text
        ▼
┌─────────────────────┐
│ B3 — Semantic Trust    │  DeBERTa-based classifier. Intent/prompt-
│ (pipeline/b3_bridge.py │  injection/semantic-manipulation detection.
│  + b3/solution_stb/)   │  Owns its own risk_level (none/low/medium/
└─────────────────────┘  high/unavailable) — see §5.
        │  SemanticResult
        ▼
┌─────────────────────┐
│ Trust Decision Engine  │  THE ONLY PLACE fusion logic exists. Combines
│ (trust_engine/)        │  B1 + B2 + B3 into one verdict.
└─────────────────────┘
        │  FinalTrustDecision
        ▼
┌─────────────────────┐
│ Adapters               │  Pure format conversion, zero trust logic.
│ (adapters/)             │  LoggingAdapter, APIAdapter, DSMassAdapter.
└─────────────────────┘
        │
        ▼
   Downstream consumer (logging, API, DS-MASS fusion, etc.)
```

**The one rule that matters most:** each layer only knows about its own
job. B1 never reasons about semantics. B2 never sees the raw payload.
B3 never does PKI validation. Only the Trust Decision Engine is allowed
to combine results into a final verdict. This is enforced by code
review convention AND by an automated test
(`tests/verify_dependency_graph.py`) that will fail CI if anyone adds an
import that violates it.

## 3. Package map — where to find/add things

| Directory | What it is | Touch this when... |
|---|---|---|
| `b1_scsv/` | Cryptographic/structural validation | You're changing what counts as a valid certificate/signature/kinematic bound |
| `b2_explain/` | Explains B1's verdict | You want to change *how* validation failures are explained to a human, or add new evidence types |
| `b3/solution_stb/b3_semantic_gate/` | The trained model + inference code | You're retraining or swapping the semantic classifier |
| `pipeline/b3_bridge.py` | Wraps the B3 model, owns `risk_level` banding | You're changing what confidence threshold counts as "high risk" semantically |
| `pipeline/synthesizer.py` | Converts raw messages → text for B3 | You're changing what facts B3 gets to see, or adding a new rendering template |
| `trust_engine/` | **The only place fusion logic lives** | You're changing how B1+B2+B3 combine into ACCEPT/CAUTION/REJECT |
| `adapters/` | Format conversion for downstream consumers | You're adding a new output format (e.g. a real DS MASS integration, a Kafka producer, etc.) |
| `pipeline/orchestrator.py` | Wires everything together (`ISCEPipeline` class) | You're changing the *order* of the pipeline, not the logic within any one stage |
| `b2_csia/` | **Legacy.** Old Dempster-Shafer trust-propagation module | Don't touch unless you're specifically archaeology-ing old behavior. It's not imported by anything live. |
| `manual_pipeline_test.py` | CLI harness for running messages/scenarios through the pipeline | You want to manually test a message or scenario |
| `tests/` | Automated test suites | Always run before pushing (see §7) |

## 4. How data actually flows (concrete types)

```python
# B1 output (from b1_scsv.scsv.SCSV.check_stateful) — normalized by the
# orchestrator into this dict shape before it goes anywhere else:
ValidationAssessment = {
    "valid": bool, "fatal": bool, "score": float, "confidence": float,
    "reasons": list[str], "checks": dict, "details": dict,
}

# B2 output (b2_explain.models.ExplainabilityReport.to_dict()):
ExplainabilityReport = {
    "explanation_text": str, "evidence": list[dict],
    "confidence_calibration": float, "provenance": dict,
    "validation_valid": bool, "validation_score": float,
}

# B3 output (pipeline.b3_bridge.SemanticResult.to_dict()):
SemanticResult = {
    "available": bool, "label": str | None, "confidence": float | None,
    "risk_level": "none" | "low" | "medium" | "high" | "unavailable",
    "status": str,
}

# Final output (trust_engine.models.FinalTrustDecision.to_dict()):
FinalTrustDecision = {
    "trust_score": float, "trust_level": "ACCEPT" | "CAUTION" | "REJECT",
    "semantic_risk": str, "cryptographic_risk": str,
    "attack_detected": bool, "confidence": float, "reasoning": str,
    "contributors": list[str], "details": dict,
}
```

**Why dicts and not just dataclass instances everywhere?** Deliberate
decision — see `ARCHITECTURE_DECISIONS.md` (AD-1). Short version: every
dict is *constructed by* a dataclass first (so it's always valid), then
flattened, because the CLI harness logs the whole pipeline result to
JSON and dataclasses aren't JSON-serializable without that step anyway.

## 5. The fusion policy (what actually decides ACCEPT/CAUTION/REJECT)

Lives in `trust_engine/decision_engine.py` + `trust_engine/policy.py`.
Five rules, most-conservative-wins:

1. **B1 fatal → REJECT**, unconditionally. B3 isn't even consulted.
2. **B1 non-fatal, score < 0.40 → REJECT** (bad crypto/validation score alone is enough).
3. **B1 non-fatal, 0.40 ≤ score < 0.70 → CAUTION**.
4. **B1 non-fatal, score ≥ 0.70**, then combined with B3's `risk_level`:
   - `risk_level = "high"` → REJECT (semantic override — this is the whole point of the paper)
   - `risk_level = "medium"` → CAUTION
   - `risk_level = "low"` → CAUTION (mild)
   - `risk_level = "none"` / `"unavailable"` → ACCEPT
5. Whenever the cryptographic-score band and the semantic-risk band
   disagree, **the lower-trust one wins.**

All thresholds are config-driven (`isce_config.yaml`), not hardcoded —
see §8.

## 6. Running it

```bash
git clone <repo>   # or unzip the delivered archive
cd intent-semantic-consistency-engine-main
pip install -r requirements.txt
```

**Single message:**
```bash
python3 manual_pipeline_test.py --pipeline test_messages/benign/normal_car.json --verbose
```

**A scenario (stateful sequence, e.g. a replay attack):**
```bash
python3 manual_pipeline_test.py --pipeline scenarios/replay/
```

**Programmatically:**
```python
from pipeline.orchestrator import ISCEPipeline
from adapters import LoggingAdapter, APIAdapter, DSMassAdapter

pipeline = ISCEPipeline(
    adapters={"log": LoggingAdapter(), "api": APIAdapter(), "ds_mass": DSMassAdapter()}
)
result = pipeline.run(messages=[msg1, msg2, msg3], context="urban")

result["decision"]                  # "ACCEPT" | "CAUTION" | "REJECT"
result["reason"]                    # human-readable explanation
result["final_trust_decision"]      # typed FinalTrustDecision dataclass
result["adapted"]["ds_mass"]        # DSMassOutput(m_A, m_not_A, m_Theta)
```

## 7. Testing — run these before every push

```bash
python3 tests/test_b2_trust_engine.py       # 29 checks: fusion rules, B2 payload isolation
python3 tests/test_adapters.py              # 25 checks: adapter purity, DS-mass math
python3 tests/verify_dependency_graph.py    # 25 checks: no cross-layer imports, no circular imports
python3 tests/verify_b3_model.py            # confirms the real model loads & predicts (needs torch installed)
```

All four also run automatically in CI (`.github/workflows/ci.yml`) on
every push/PR, across Python 3.10–3.12. The dependency-graph check in
particular will **fail your PR** if you accidentally add an import that
breaks layer isolation (e.g. B1 importing from `trust_engine`) — that's
intentional, don't work around it, fix the import instead.

## 8. Configuration — `isce_config.yaml`

```yaml
b3_semantic_gate:
  model_path: "b3/solution_stb/b3_semantic_gate/model/semantic_gate_v3"
  risk_thresholds:
    high: 0.85     # B3 confidence >= this on a malicious label -> risk_level="high"
    medium: 0.60   # >= this (below high) -> "medium"; below "medium" -> "low"

trust_engine:
  semantic_high_confidence: 0.85      # fallback only, see ARCHITECTURE_DECISIONS.md AD-2
  semantic_medium_confidence: 0.60
  cryptographic_reject_below: 0.40
  cryptographic_caution_below: 0.70

b2_explain:
  check_descriptions: {}              # override/extend B1-check → human-readable-text mappings
  sparse_evidence_confidence_cap: 0.5
```

Never hardcode a threshold in code. If you need a new one, add it to
this file and read it via the relevant `*Policy`/`*Config`
`.from_config()` static method (see `trust_engine/policy.py` or
`b2_explain/config.py` for the pattern).

## 9. What's NOT done yet — be aware before you present/publish anything

1. **B3's real model inference has not been verified on live hardware
   by Claude** — only structurally (paths resolve, config loads,
   graceful degradation works) because the sandbox it was built in had
   no network access to install `torch`/`transformers`. **Someone on
   the team needs to run `tests/verify_b3_model.py` on a real machine
   and confirm it prints real predictions**, not the stub fallback.
2. **`adapters/ds_mass_adapter.py`'s exact field mapping
   (`m_A = trust_score * confidence`, etc.) is a documented, defensible
   construction — not validated against whatever the actual downstream
   DS MASS consumer expects.** If someone owns that integration, they
   should review this one file and adjust the formula if needed; it's
   isolated to a single file by design.
3. **CI has never executed on real GitHub infrastructure** — it was
   only simulated locally. First PR should confirm it actually runs green.
4. **`b2_csia/` is legacy and unused**, but its own standalone test
   suite (`b2_csia/test_*.py`) still exists and still passes on its own
   — it's just disconnected from the live pipeline. Don't be confused
   if you see it; don't delete it without a separate conversation.
5. **`validation/test_pipeline_equivalence.py`** asserts an old
   invariant (`ISCEPipeline` output == direct legacy `CSIA` computation)
   that is now intentionally false, since B2's role changed by design
   from "trust propagation" to "explainability." It should be marked
   `@skip` with a note, not "fixed" — fixing it would mean reverting B2.

## 10. Key documents in the repo

- `ARCHITECTURE_DECISIONS.md` — why specific design choices were made
  (dict-vs-dataclass boundary, B3 owning `risk_level`, adapter purity),
  written so they read as decisions, not unexplained inconsistencies.
- `INTEGRATION_README.md` — the fuller technical integration log:
  what changed, what was verified, what bugs were found and fixed along
  the way. Good background reading if you want the "how did we get
  here" story, not just the current state.
- This document — onboarding / how to work in this repo day-to-day.

## 11. Quick sanity check for a new contributor's first day

```bash
pip install -r requirements.txt
python3 tests/test_b2_trust_engine.py && \
python3 tests/test_adapters.py && \
python3 tests/verify_dependency_graph.py && \
python3 manual_pipeline_test.py --pipeline test_messages/benign/normal_car.json --verbose
```
If all of that runs clean, your environment is set up correctly and
you're ready to make changes.
