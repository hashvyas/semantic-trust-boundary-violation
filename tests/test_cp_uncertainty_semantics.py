"""
tests/test_cp_uncertainty_semantics.py
=========================================
Regression tests for the corrected CP evidence semantics (see
CHANGELOG.md): low cooperative-perception CORROBORATION must raise
Dempster-Shafer uncertainty (Theta mass), while genuine inter-report
CONTRADICTION must still contribute disbelief (not_A mass).

Mathematical grounding (see also pipeline/orchestrator.py's fold comment
and trust_engine/dempster_shafer.py's module docstring):
- Shafer, "A Mathematical Theory of Evidence" (1976): ignorance is mass
  committed to the whole frame Theta; absence of evidence for A is NOT
  evidence for not_A.
- Yager, "On the Dempster-Shafer framework and new combination rules",
  Information Sciences 41(2), 1987: conflict mass reassigned to Theta,
  which is why Theta-routed sparsity propagates as uncertainty rather
  than being normalized away.

Tests drive the REAL orchestrator fold + REAL TrustDecisionEngine; only
_run_cp is stubbed to supply controlled CPResult dicts (interface-shaped,
same keys cp_layer.cp_layer() emits).

Run with: python3 tests/test_cp_uncertainty_semantics.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from b1_scsv.scsv import SCSV
from pipeline.orchestrator import ISCEPipeline
from adapters import LoggingAdapter, APIAdapter, DSMassAdapter

_FAILURES = []


def check(name, cond, evidence=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {evidence}" if evidence else ""))
    if not cond:
        _FAILURES.append(name)


def make_pipe(cp_result, enable_mbd=True):
    pipe = ISCEPipeline(
        scsv=SCSV(cert_rotation_owner="mbd"), enable_mbd=enable_mbd, enable_cp=True, pki_ca=None,
        adapters={"log": LoggingAdapter(), "api": APIAdapter(), "ds_mass": DSMassAdapter()},
    )
    if cp_result is not None:
        pipe._run_cp = lambda messages, target_sender_weight, target_sender_id: dict(cp_result)
    return pipe


def cp_result(spatial=1.0, speed=1.0, heading=1.0, diversity=1.0, num_reports=4,
              event_label=None):
    conf = spatial * 0.35 + speed * 0.25 + heading * 0.20 + diversity * 0.20
    return {"boundary": "CP", "event_label": event_label, "num_reports": num_reports,
            "senders": list(range(num_reports)), "spatial_score": spatial,
            "speed_score": speed, "heading_score": heading, "diversity_score": diversity,
            "cp_confidence": round(conf, 3), "fusion_confidence": round(conf, 3),
            "cp_pass": conf > 0.7, "reports": []}


BENIGN = json.loads((ROOT / "test_messages" / "benign" / "normal_car.json").read_text())

print("=" * 78)
print("CP UNCERTAINTY-vs-DISBELIEF SEMANTICS")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1. LOW CORROBORATION (reports agree, but few/low-diversity independent
#    sources): must raise Theta mass -> CAUTION at most, never REJECT, and
#    never manufacture disbelief.
# ---------------------------------------------------------------------------
print("\n--- 1. Low corroboration -> uncertainty (Theta), not rejection ---")
r_sparse = make_pipe(cp_result(spatial=1.0, speed=1.0, heading=1.0,
                                 diversity=0.15, num_reports=4)).run([BENIGN], context="urban")
fd = r_sparse["fusion"]
mass = fd["details"]["ds_crypto_mass"]
check("Decision is not REJECT under pure corroboration deficit",
      r_sparse["decision"] != "REJECT", f"decision={r_sparse['decision']}")
check("Decision degrades to CAUTION (caution-until-corroborated design intent)",
      r_sparse["decision"] == "CAUTION", f"decision={r_sparse['decision']}")
check("Theta mass increased (confidence lowered by corroboration deficit)",
      mass["m_theta"] > 0.5, f"m_theta={mass['m_theta']:.3f}")
check("No disbelief manufactured from sparsity (m_not_A ~ 0)",
      mass["m_not_A"] < 0.05, f"m_not_A={mass['m_not_A']:.3f}")
check("attack_detected is False under pure sparsity", fd["attack_detected"] is False)

# Contrast: well-corroborated agreeing cluster -> ACCEPT. MBD disabled here
# to isolate CP's contribution: with MBD enabled, a FRESH sender already
# lowers confidence_calibration by design (confirmed intended
# caution-until-corroborated behavior), which would mask CP's effect.
r_dense = make_pipe(cp_result(diversity=0.9), enable_mbd=False).run([BENIGN], context="urban")
check("Well-corroborated agreeing cluster -> ACCEPT (CP isolated, MBD off)",
      r_dense["decision"] == "ACCEPT", f"decision={r_dense['decision']}")
check("Theta mass low when corroboration high",
      r_dense["fusion"]["details"]["ds_crypto_mass"]["m_theta"]
      < mass["m_theta"], "monotone in corroboration")

# ---------------------------------------------------------------------------
# 2. GENUINE CONTRADICTION (independent reports exist and actively disagree
#    spatially/kinematically): must still contribute disbelief -> REJECT.
# ---------------------------------------------------------------------------
print("\n--- 2. Genuine contradiction -> disbelief (not_A), rejection ---")
r_conflict = make_pipe(cp_result(spatial=0.1, speed=0.2, heading=0.2,
                                   diversity=0.9, num_reports=5, event_label="obstacle")).run([BENIGN], context="urban")
fd_c = r_conflict["fusion"]
mass_c = fd_c["details"]["ds_crypto_mass"]
check("Decision is REJECT under genuine inter-report contradiction",
      r_conflict["decision"] == "REJECT", f"decision={r_conflict['decision']}")
check("Disbelief dominates belief under contradiction (m_not_A >> m_A)",
      mass_c["m_not_A"] > 3 * mass_c["m_A"],
      f"m_not_A={mass_c['m_not_A']:.3f} vs m_A={mass_c['m_A']:.3f}")
# Absolute-dominance form, CP isolated (MBD's intended fresh-sender
# confidence damping otherwise caps total committed mass):
mass_ci = make_pipe(cp_result(spatial=0.1, speed=0.2, heading=0.2, diversity=0.9,
                                num_reports=5, event_label="obstacle"), enable_mbd=False) \
    .run([BENIGN], context="urban")["fusion"]["details"]["ds_crypto_mass"]
check("Disbelief mass dominates absolutely when CP isolated (m_not_A > 0.5, MBD off)",
      mass_ci["m_not_A"] > 0.5, f"m_not_A={mass_ci['m_not_A']:.3f}")
check("attack_detected is True under contradiction", fd_c["attack_detected"] is True)

# Mixed case: moderate disagreement, moderate corroboration -> disbelief AND
# theta both present; decision at least CAUTION.
r_mixed = make_pipe(cp_result(spatial=0.6, speed=0.6, heading=0.6,
                                diversity=0.5, num_reports=4, event_label="obstacle")).run([BENIGN], context="urban")
check("Mixed disagreement+deficit -> at least CAUTION",
      r_mixed["decision"] in ("CAUTION", "REJECT"), f"decision={r_mixed['decision']}")

# Geometry-vs-contradiction distinction: spatially spread PLAIN CAM traffic
# (no shared claimed event) must not be treated as contradiction, even with
# spatial_score = 0 -- that is exactly the measured FPR-0.93 failure mode.
r_spread = make_pipe(cp_result(spatial=0.0, speed=0.75, heading=0.95,
                                 diversity=1.0, num_reports=8, event_label=None),
                      enable_mbd=False).run([BENIGN], context="urban")
# Baseline WITHOUT CP on the same message: any disbelief present there
# (e.g. B1's legitimate fixture-staleness penalty, audit item S1) is not
# CP's doing; the assertion is that CP ADDS none.
def main():
    print("=" * 78)
    print("CP UNCERTAINTY-vs-DISBELIEF SEMANTICS")
    print("=" * 78)

    # ---------------------------------------------------------------------------
    # 1. LOW CORROBORATION (reports agree, but few/low-diversity independent
    #    sources): must raise Theta mass -> CAUTION at most, never REJECT, and
    #    never manufacture disbelief.
    # ---------------------------------------------------------------------------
    print("\n--- 1. Low corroboration -> uncertainty (Theta), not rejection ---")
    r_sparse = make_pipe(cp_result(spatial=1.0, speed=1.0, heading=1.0,
                                   diversity=0.0, num_reports=2)).run([BENIGN], context="urban")
    m_sparse = r_sparse["adapted"]["ds_mass"]
    check("Sparse reports (diversity=0, reports=2) -> CAUTION, not REJECT",
          r_sparse["decision"] == "CAUTION", f"got={r_sparse['decision']}")
    check("Sparse reports yield elevated Theta (ignorance) mass",
          m_sparse.m_Theta > 0.40, f"m_Theta={m_sparse.m_Theta:.4f}")
    check("Sparse reports do NOT manufacture disbelief (m_not_A is zero)",
          m_sparse.m_not_A < 1e-9, f"m_not_A={m_sparse.m_not_A:.4f}")

    # ---------------------------------------------------------------------------
    # 2. INTER-REPORT CONTRADICTION: genuine disagreement on kinematics/events:
    #    must result in REJECT or CAUTION via disbelief (not_A mass), not merely
    #    ignorance.
    # ---------------------------------------------------------------------------
    print("\n--- 2. Inter-report contradiction -> disbelief (not_A), can reject ---")
    r_conflict = make_pipe(cp_result(spatial=0.0, speed=0.0, heading=0.0,
                                     diversity=1.0, num_reports=4)).run([BENIGN], context="urban")
    m_conflict = r_conflict["adapted"]["ds_mass"]
    check("Contradictory reports (scores=0, reports=4) -> REJECT",
          r_conflict["decision"] == "REJECT", f"got={r_conflict['decision']}")
    check("Contradictory reports commit significant disbelief (not_A) mass",
          m_conflict.m_not_A > 0.40, f"m_not_A={m_conflict.m_not_A:.4f}")

    # ---------------------------------------------------------------------------
    # 3. SINGLE-REPORT WINDOW: CP is silent, not maximally ignorant.
    #    When num_reports=1 (no peer data), CP is skipped. The final decision is
    #    determined by SCSV + MBD alone; CP does not inject artificial ignorance to
    #    degrade their trust score.
    # ---------------------------------------------------------------------------
    print("\n--- 3. Single-report window: CP is silent, not maximally ignorant ---")
    r_single = make_pipe(cp_result(diversity=0.0, num_reports=1)).run([BENIGN], context="urban")
    r_nocp = ISCEPipeline(scsv=SCSV(cert_rotation_owner="mbd"), enable_mbd=True,
                            enable_cp=False,
                            adapters={"log": LoggingAdapter(), "api": APIAdapter(),
                                       "ds_mass": DSMassAdapter()}).run([BENIGN], context="urban")
    check("num_reports=1: decision matches CP-disabled pipeline (no phantom ignorance)",
          r_single["decision"] == r_nocp["decision"],
          f"single={r_single['decision']} nocp={r_nocp['decision']}")

    print()
    print("=" * 78)
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S): {_FAILURES}")
        sys.exit(1)
    print("All CP uncertainty-vs-disbelief semantics checks passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
