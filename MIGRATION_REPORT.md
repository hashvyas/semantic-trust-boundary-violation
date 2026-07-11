# Migration Report — PKI / MBD / CP Integration

**Architecture status: frozen and implemented as specified**, with two
scoping limitations documented honestly below (not silently glossed
over) and one additional duplication found and fixed during
implementation that wasn't in the original audit.

---

## 1. What was implemented

### PKI (`pki/`)
- Integrated from `modules/pki.py`. `"boundary"` renamed `"B1_PKI"` → `"PKI"`.
- Added `compromised`/`revoked` as explicit top-level result fields (previously only inspectable via the raw cert dict) — needed so MBD/B2/TrustEngine can see the "compromised but not revoked" state without reaching into PKI's certificate representation.
- Wired into `ISCEPipeline` as **opt-in** (`pki_ca=` constructor param). Only runs if a message carries `_pki_signature`/`_pki_certificate`/`_pki_public_key`; otherwise skipped, not faked as passing. See §4 for why.

### B1 (`b1_scsv/scsv.py`)
- Added `cert_rotation_owner: Literal["b1","mbd"] = "b1"` constructor parameter. Default preserves **exact** pre-existing behavior (verified byte-identical in testing).
- Added `check_cert_rotation_for_station()` public accessor so MBD can query B1's own `VehicleStateManager` — the cert-rotation algorithm now lives in exactly one place (audit finding D1's resolution), not duplicated.
- Zero other changes to B1's logic.

### MBD (`mbd/`)
- Integrated from `modules/mbd.py`. `boundary` renamed `"B2_MBD"` → `"MBD"`.
- Operates on the flat `{sender, x, y, speed, heading, timestamp, event}` schema — **not** raw CAM messages (see `bridges/message_adapter.py`, §3).
- `certificate_rotation_score()` added, consuming B1's delegated signal (via `check_cert_rotation_for_station`) rather than reimplementing rotation tracking.
- Wired as **opt-in** (`enable_mbd=True`). Runs between B1 and B2; when enabled, B2 uses the new `explain_evidence()` path instead of `explain()`.
- Removed the uploaded module's debug `print()` statements (not production-quality).

### B2 (`b2_explain/explainability.py`)
- Added `explain_evidence(evidence: List[TrustEvidence])` — **additive**, existing `explain()` untouched. Combines B1 + MBD (and CP, when wired) into one unified report without modifying any upstream verdict.

### CP (`cp/`)
- Integrated from `modules/cp_layer.py` **only**. `modules/cp.py` (confirmed via `diff` to be a strict subset/earlier draft, audit finding D4) was **not carried into this repository** — verified by `tests/test_pki_mbd_cp_integration.py`'s explicit `**/cp.py` search, which asserts zero matches.
- `boundary` renamed `"B3_CP"` → `"CP"`.
- Wired as **opt-in** (`enable_cp=True`). Runs between B2 and CP, consuming `observation_weights` derived from B2's evidence (see §4 for the scoping limitation on per-peer weighting).

### `contracts/` (new)
- `TrustEvidence` dataclass — the common lens B2 reads across B1/MBD/CP. Deliberately placed in a **neutral, zero-dependency package**, not `trust_engine/` — B2 is architecturally forbidden from importing `trust_engine` (mechanically enforced), so a shared contract needed a home neither B2 nor Trust Engine "owns."

### `bridges/` (new)
- `message_adapter.py` — converts this repo's nested ETSI-fixed-point CAM messages into MBD/CP's flat, meter-based schema. This is **load-bearing, not cosmetic**: MBD's Sybil check uses `dist < 2.0` (meters), CP's spatial consistency divides by `20` (meters) — feeding raw lat/lon integers directly would silently corrupt every score. Uses an equirectangular projection (documented as accurate for the tens-to-hundreds-of-meters ranges these thresholds are calibrated for, not for city-scale distances).
- Raises `ValueError` rather than silently defaulting missing fields — a silently-defaulted position would corrupt every downstream MBD/CP score.

### `simulation/` (renamed from `modules/decision_engine.py`, `modules/llmagent.py`)
- `llm_dispatcher.py` — per audit finding D5, renamed to avoid the `decision_engine.py` naming collision with `trust_engine/decision_engine.py`.
- **One additional duplication found and fixed during this integration** (not in the original audit — found while actually porting the code): the uploaded `decision_engine()` directly re-checked `pki_result["pki_pass"]` / `mbd_result["mbd_pass"]` and returned `"REJECT"` itself. This duplicates trust-fusion logic the Trust Decision Engine already owns exclusively. **Fixed:** `llm_dispatcher()` now reads `FinalTrustDecision.trust_level` as its sole source of REJECT information; it never re-inspects PKI/B1/MBD/CP results directly. The two genuine bug fixes already present in the uploaded version (TTC computed regardless of confidence gate; explicit `distance_missing` handling instead of a silently-optimistic default) are preserved unchanged, since those are legitimate downstream-decision logic, not trust fusion.

---

## 2. Validation results (all re-run this session, not carried over from memory)

| Suite | Result |
|---|---|
| `tests/test_b2_trust_engine.py` | 29/29 PASS |
| `tests/test_adapters.py` | 25/25 PASS |
| `tests/verify_dependency_graph.py` | 39/39 PASS (extended with PKI/MBD/CP/contracts/bridges/simulation isolation checks) |
| `tests/test_pki_mbd_cp_integration.py` (new) | 30/30 PASS |
| `manual_pipeline_test.py --pipeline` fixture/scenario sweep | Identical results to pre-integration baseline across all 8 categories — **zero regressions** |

**Total: 84 targeted assertions + full fixture/scenario regression, all passing.**

### Duplication verification (per your explicit "verify no duplicate functionality remains" instruction)
- `**/cp.py` search: zero matches anywhere in the repository (D4 resolved).
- `"B1_PKI"`/`"B2_MBD"`/`"B3_CP"` string-literal search across `pki/`, `mbd/`, `cp/`: zero live occurrences outside rename-explaining comments (§0 resolved).
- AST-level check: `b1_scsv/scsv.py` contains zero references to `VehicleHistoryStore` and zero imports of `mbd` (D3's single-message/multi-message split mechanically enforced, not just documented).
- Cert-rotation algorithm: confirmed to exist in exactly one place (`b1_scsv.scsv._VehicleStateManager.check_cert_rotation`); MBD delegates via `check_cert_rotation_for_station` rather than reimplementing (D1 resolved).
- Replay detection: confirmed **intentionally** dual (B1's exact-match structural cache + MBD's behavioral pattern score), both independently verified to fire on the same replay scenario — this is the audit's documented resolution for D2 (complementary, not merged), not an oversight.
- LLM Dispatcher: confirmed it no longer re-implements REJECT logic — traced to read only `FinalTrustDecision.trust_level`.

---

## 3. Dependency graph (final, mechanically verified)

```
pki/            -> (stdlib, cryptography) only
b1_scsv/        -> b1_scsv.* only (unchanged)
mbd/            -> (stdlib) only -- no import of b1_scsv, despite the
                    orchestrator-level delegation for cert rotation
cp/             -> numpy only
contracts/      -> (stdlib) only -- leaf module
bridges/        -> (stdlib: math) only -- leaf module
b2_explain/     -> b2_explain.*, contracts.trust_evidence only
trust_engine/   -> trust_engine.* only (unchanged)
adapters/       -> trust_engine.models only (unchanged)
simulation/     -> (stdlib) only -- reads FinalTrustDecision as a
                    plain object via getattr, no import of trust_engine
                    or any upstream layer required
pipeline/orchestrator.py -> imports across all layers (composition root, correct)
```

No circular imports (39/39 dependency graph checks pass, including live import of every module).

---

## 4. Scoping limitations — documented, not silently glossed over

1. **CP's `observation_weights` are only fully trust-weighted for the target message's sender.** Other senders in a message window default to weight `1.0` (unweighted), because `ISCEPipeline.run()` validates one target message per call through B1+MBD, not every peer independently. Full per-peer trust-weighted CP (running B1+MBD for every report in the window before fusion) is a further increment, not implemented here — it would require restructuring `run()` to accept and validate a full batch, which is a larger change than "wire CP in" and deserves its own review.

2. **PKI requires messages to carry actual signature/certificate/public-key material to do anything.** None of this repo's existing `test_messages/*.json` fixtures carry that material (they only have a `certificate_id` string, not real cryptographic payloads) — this is a pre-existing gap in the fixture set, not something this integration could silently paper over. PKI is wired and tested (5/5 unit tests passing against `pki.CertificateAuthority`-issued certs), but has not been exercised against the repository's actual message fixtures because none of them are signed. Generating signed fixtures is a follow-up task, not attempted here.

3. **Full MBD/CP wiring is opt-in, not the new default**, per your "preserve backward compatibility" requirement from the original architecture brief. `manual_pipeline_test.py` and all existing callers see zero behavior change unless they explicitly pass `enable_mbd=True`/`enable_cp=True`/`pki_ca=...`.

---

## 5. Files added/modified — final list with locations

**New packages:**
- `pki/__init__.py`, `pki/pki_layer.py`
- `mbd/__init__.py`, `mbd/mbd_layer.py`
- `cp/__init__.py`, `cp/cp_layer.py`
- `contracts/__init__.py`, `contracts/trust_evidence.py`
- `bridges/__init__.py` *(implicit — see note)*, `bridges/message_adapter.py`
- `simulation/__init__.py`, `simulation/llm_dispatcher.py`

**Modified:**
- `b1_scsv/scsv.py` — added `cert_rotation_owner` param, `check_cert_rotation_for_station()` accessor. Zero change to default-mode behavior (verified).
- `b2_explain/explainability.py` — added `explain_evidence()`. Zero change to existing `explain()`.
- `pipeline/orchestrator.py` — added PKI/MBD/CP wiring, all opt-in. Zero change to default-mode behavior (verified).
- `tests/verify_dependency_graph.py` — extended with 14 new checks covering the new packages.

**New tests:**
- `tests/test_pki_mbd_cp_integration.py` — 30 checks covering PKI/MBD/CP units, D1-D5 resolutions, and backward-compatibility.

Note: `bridges/__init__.py` was initially missing (the package worked
anyway via Python 3's implicit namespace-package support), caught and
fixed during this report's own verification pass so `bridges/` is
consistent with every other package in the repo (explicit `__init__.py`,
not relying on implicit namespace packaging).
