# ISCE Pipeline — B1 / B2 / B3 / Trust Engine Integration

## Final architecture

```
Incoming V2X Message
        │
        ▼
┌────────────────────────────┐
│ B1 — b1_scsv (SCSV)         │  Cryptographic / structural validation
│ unchanged                   │  (cert, signature, revocation, replay,
└────────────────────────────┘   kinematics, timestamp freshness)
        │  ValidationAssessment (dict: valid, fatal, score, confidence,
        │                        reasons, checks, details)
        ▼
┌────────────────────────────┐
│ B2 — b2_explain              │  Explains B1's verdict ONLY.
│ ExplainabilityEngine         │  Input: ValidationAssessment dict ONLY.
└────────────────────────────┘  Never sees the raw message. Never does
        │  ExplainabilityReport   semantic reasoning. Never overrides B1.
        ▼
┌────────────────────────────┐
│ Synthesizer                 │  Builds B3's input text from the RAW
│ pipeline/synthesizer.py     │  message cluster only (never from B2 —
│ unchanged                   │  b2_result param accepted, never read).
└────────────────────────────┘
        │  scene text
        ▼
┌────────────────────────────┐
│ B3 — b3_bridge /             │  Semantic classification (DeBERTa model,
│ b3_semantic_gate             │  falls back to StubSemanticClassifier
│                               │  when no checkpoint present). Owns
└────────────────────────────┘  risk_level -- its own public contract.
        │  SemanticResult {available, label, confidence, risk_level, status}
        ▼
┌────────────────────────────┐
│ Trust Decision Engine         │  ONLY component that fuses B1+B2+B3.
│ trust_engine/decision_engine  │  Policy-driven, configurable thresholds.
└────────────────────────────┘
        │  FinalTrustDecision (typed dataclass)
        ▼
┌────────────────────────────┐
│ Adapters                     │  Pure format conversion. ZERO trust
│ adapters/*.py                │  logic (mechanically verified in
└────────────────────────────┘  tests/test_adapters.py via AST import check).
        │
        ├──► LoggingAdapter  → structured log record dict
        ├──► APIAdapter      → versioned JSON API envelope
        └──► DSMassAdapter   → Dempster-Shafer mass function (m_A, m_not_A, m_Theta)
                                (matches b2_csia.uncertainty.MassFunction convention)
```

`b2_csia/` (old Dempster-Shafer/trust-propagation module) is **untouched
and unimported** by the pipeline — kept as legacy for reference, not
deleted, per your "don't remove abstractions unnecessarily" constraint.

See `ARCHITECTURE_DECISIONS.md` for the reasoning behind specific design
choices (dict-as-wire-format, B3 owning `risk_level`, adapter purity).

## Fusion policy (trust_engine/policy.py)

1. **B1 fatal → REJECT**, unconditionally. B3 is not consulted.
2. **B1 non-fatal, score < 0.40 → REJECT** (cryptographic risk band).
3. **B1 non-fatal, 0.40 ≤ score < 0.70 → CAUTION**.
4. **B1 non-fatal, score ≥ 0.70**, then layered with B3:
   - B3 label=MALICIOUS, confidence ≥ 0.85 → **REJECT**
   - B3 label=MALICIOUS, 0.60 ≤ confidence < 0.85 → **CAUTION**
   - B3 label=MALICIOUS, confidence < 0.60 → **CAUTION** (mild)
   - B3 benign / unavailable → **ACCEPT**
5. Whenever both the cryptographic score band and semantic risk band
   disagree, the **more conservative (lower-trust) of the two wins**.

All thresholds are overridable via `isce_config.yaml` under a
`trust_engine:` key (see `TrustPolicy.from_config`).

## New files

```
b2_explain/
    __init__.py
    models.py            EvidenceItem, ExplainabilityReport
    explainability.py    ExplainabilityEngine.explain(validation_assessment)

trust_engine/
    __init__.py
    models.py             FinalTrustDecision, TrustLevel, SemanticRisk
    policy.py              TrustPolicy (configurable thresholds)
    decision_engine.py     TrustDecisionEngine.decide(b1, b2, b3)

tests/
    test_b2_trust_engine.py   29 checks: fusion rules, B2 payload isolation.
    test_adapters.py          25 checks: adapter purity, dependency
                               isolation (AST-verified), DS mass math.
    verify_dependency_graph.py 25 checks: mechanically verifies every
                               cross-layer import constraint + no
                               circular imports (not just claimed).
    verify_b3_model.py        Loads the real B3 model and runs 2 sample
                               predictions.

adapters/
    __init__.py
    base.py               Adapter ABC -- the zero-trust-logic contract.
    logging_adapter.py     FinalTrustDecision -> structured log dict.
    api_adapter.py          FinalTrustDecision -> versioned JSON API envelope.
    ds_mass_adapter.py      FinalTrustDecision -> Dempster-Shafer mass
                            function (m_A, m_not_A, m_Theta), compatible
                            with b2_csia.uncertainty.MassFunction's
                            existing convention.

.github/workflows/ci.yml   Runs both test suites + dependency graph
                            check + full fixture/scenario regression
                            sweep on every push, across Python 3.10-3.12.

b3/solution_stb/b3_semantic_gate/model/semantic_gate_v3/
    Trained DeBERTa-v2 checkpoint (from B3_model_v3.zip). ~567MB.
```

## Modified files

- `pipeline/orchestrator.py` — rewired to call `ExplainabilityEngine`
  instead of `CSIA`, and `TrustDecisionEngine` instead of
  `pipeline/fusion.py`. B2 now runs even on the B1-fatal path (previously
  bypassed), so a REJECT always comes with an explanation. Accepts a
  deprecated `csia=` kwarg for backward compatibility with existing call
  sites (`manual_pipeline_test.py`, `validation/test_pipeline_equivalence.py`)
  — the value is stored but unused.

## Not modified (verified compatible, no changes needed)

- `b1_scsv/*` — untouched.
- `pipeline/synthesizer.py` — already never reads B2 output; its
  `b2_result` parameter was already a no-op placeholder for API stability.
- `pipeline/b3_bridge.py` — untouched.

## Two real bugs found and fixed during integration testing

1. **Contributor list leaked B3 into a fatal-path decision** even though
   B3 is explicitly not consulted when B1 fails fatally. Fixed by
   returning `contributors=["B1","B2"]` unconditionally on the fatal path.
2. **Operator-precedence bug** in the cryptographic score banding
   (`... or not b1_valid and ...`) caused any non-fatal-but-invalid
   message with score < 0.70 to always REJECT instead of following the
   intended REJECT(<0.40)/CAUTION(<0.70)/ACCEPT(≥0.70) bands. Fixed by
   making the bands purely score-driven.

Both were caught by `tests/test_b2_trust_engine.py`, which failed before
the fix and passes after — proof the regression is closed, not just
patched blind.

## B3 trained model — now included

`b3/solution_stb/b3_semantic_gate/model/semantic_gate_v3/` contains the
trained DeBERTa-v2 checkpoint (6-layer, `BENIGN` / `MALICIOUS_SEMANTIC_MANIPULATION`
labels — matches `pipeline/b3_bridge.py`'s label mapping exactly, no code
changes needed there). This sandbox has no network access and could not
install `torch`/`transformers` to run live inference, so this was
integrated and verified structurally (model path resolution, config
loading, graceful-degradation behavior) but **not** verified with actual
forward-pass inference. Run `tests/verify_b3_model.py` on your machine
after `pip install -r requirements.txt` to confirm real inference works.

Note: `B3_for_team.rar` could not be extracted (no unrar/7z in this
sandbox, no network to install one) and was skipped per your instruction.
If it contains additional modules beyond the model checkpoint, re-upload
as `.zip` (of the extracted contents, not the `.rar` itself) in a future
turn and I'll integrate it.

### Install and verify

```bash
pip install -r requirements.txt
python3 tests/verify_b3_model.py
```

Expected output: two test predictions (one benign scene, one
prompt-injection-style payload) with `available: True` and real
label/confidence values instead of the stub's `available: False`.

## How to use it

### 0. Install dependencies

```bash
cd intent-semantic-consistency-engine-main
pip install -r requirements.txt
```

### 1. Run a single message through the full pipeline

```bash
cd intent-semantic-consistency-engine-main
python3 manual_pipeline_test.py --pipeline test_messages/benign/normal_car.json --verbose
```

### 2. Run a directory / scenario as a stateful sequence

```bash
python3 manual_pipeline_test.py --pipeline scenarios/replay/
python3 manual_pipeline_test.py --pipeline scenarios/sybil/
```

### 3. Run the new regression test suite (no pytest needed)

```bash
python3 tests/test_b2_trust_engine.py
```

### 4. Use it programmatically

```python
from pipeline.orchestrator import ISCEPipeline
from adapters import LoggingAdapter, APIAdapter, DSMassAdapter

pipeline = ISCEPipeline(
    adapters={"log": LoggingAdapter(), "api": APIAdapter(), "ds_mass": DSMassAdapter()}
)
result = pipeline.run(messages=[msg1, msg2, msg3], context="urban")

result["decision"]              # "ACCEPT" | "CAUTION" | "REJECT"
result["reason"]                # human-readable reasoning string
result["b2"]["explanation_text"]  # B2's explanation of B1's verdict
result["final_trust_decision"]  # the typed FinalTrustDecision dataclass
result["adapted"]["ds_mass"]    # DSMassOutput(m_A, m_not_A, m_Theta) for DS MASS
result["adapted"]["api"]        # versioned JSON API envelope
result["adapted"]["log"]        # structured log record dict
```

### 5. Tune fusion thresholds without touching code

Add to `isce_config.yaml`:
```yaml
trust_engine:
  semantic_high_confidence: 0.90
  semantic_medium_confidence: 0.65
  cryptographic_reject_below: 0.35
  cryptographic_caution_below: 0.75
```
Then: `TrustPolicy.from_config(yaml.safe_load(open("isce_config.yaml")))`.

## Known pre-existing issue (not introduced by this change)

Running `manual_pipeline_test.py --pipeline test_messages/b1_fail/`
(the *directory* form) fails with a timestamp sort error — that
directory intentionally mixes fixtures with malformed/non-numeric
timestamps (`malformed.json`, `certificate_switch.json`, `replay.json`)
for *single-file* testing, which breaks the harness's directory-mode
`sorted(..., key=get_timestamp)` call. This exists independent of any
change made here — confirmed by checking the raw fixture data types.
Running each file individually works correctly.

## What still remains out of scope of this pass

- `b2_csia` is legacy and unused by the pipeline now, but its own test
  suite (`b2_csia/test_*.py`) and `validation/run_phase2_*` scripts still
  exist standalone — they weren't touched and still pass on their own,
  since they never went through `ISCEPipeline`.
- `validation/test_pipeline_equivalence.py` asserts the *old* invariant
  that `ISCEPipeline` output equals direct `CSIA` computation. That
  invariant is now intentionally false — B2's semantics changed by
  design (explainability, not trust-propagation). Recommend marking that
  test deprecated/skipped rather than "fixing" it, since fixing it would
  mean reintroducing the old B2 behavior.
