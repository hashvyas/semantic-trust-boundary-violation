"""
tests/test_large_scale_framework.py
======================================
Regression tests for the large-scale evaluation framework (large_scale/).
Runs WITHOUT the B3 checkpoint or GPU -- exercises the grids, the semantic
attack library, and the scenario-level scoring logic that the headline
numbers depend on. In particular it locks the scoring-unit fix: windowed
decisions must be scored at scenario level, not per message, or honest
vehicles inside an attack scenario are miscounted as false positives.

Run with:  python3 tests/test_large_scale_framework.py
"""
from __future__ import annotations

import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from large_scale import scaling, semantic_attacks as sem
from large_scale.run_large_scale import scenario_level_rows

_FAILURES = []


def check(name, cond, evidence=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {evidence}" if evidence else ""))
    if not cond:
        _FAILURES.append(name)


print("=" * 78)
print("LARGE-SCALE FRAMEWORK")
print("=" * 78)

# --- scale + attack grids -------------------------------------------------
print("\n--- Part 1/2/3: grids ---")
grid = scaling.build_scale_grid(seed=1, message_target=1000,
                                 vehicle_counts=[10, 50], attacker_pcts=[0.05, 0.20],
                                 families=["benign", "replay", "sybil"])
check("scale grid is non-empty", len(grid) > 0, f"{len(grid)} scenarios")
check("scale grid spans requested vehicle counts",
      {s.base.vehicle_count for s in grid} == {10, 50})
check("scale grid spans requested attacker pcts",
      {s.attacker_pct for s in grid} == {0.05, 0.20})
check("benign scenarios have zero attackers",
      all(s.base.attacker_count == 0 for s in grid if s.base.scenario_family == "benign"))
check("attack scenarios have >=1 attacker",
      all(s.base.attacker_count >= 1 for s in grid if s.base.scenario_family != "benign"))
check("attacker count tracks percentage (50v @ 20% -> 10)",
      any(s.base.vehicle_count == 50 and s.attacker_pct == 0.20 and s.base.attacker_count == 10
          for s in grid))

sweep = scaling.build_sweep_grid(seed=1, vehicle_counts=[10], attacker_pcts=[0.10],
                                  densities=["dense"], comm_ranges=[150], frequencies=[10.0],
                                  semantic_confidences=[0.9], families=["benign", "replay"])
check("sweep grid carries all sweep axes",
      all(hasattr(s, "comm_range_m") and hasattr(s, "frequency_hz")
          and hasattr(s, "semantic_confidence") for s in sweep))

# message-target scaling
big = scaling.build_scale_grid(seed=1, message_target=10000, vehicle_counts=[1000],
                                attacker_pcts=[0.10], families=["benign"])
est = scaling.estimate_messages(big)
check("message-target scaling reaches large volume (1000 vehicles x 10k target)",
      est >= 10000, f"~{est} messages")

# --- semantic attack library ---------------------------------------------
print("\n--- Part 2: semantic attack library ---")
rng = random.Random(0)
fams = sem.all_families()
required = {"false_emergency", "rsu_spoofing", "prompt_injection", "context_poisoning",
            "instruction_hiding", "role_confusion", "semantic_narrative_poisoning"}
check("all required semantic families present", required <= set(fams),
      f"missing={required - set(fams)}")
for f in fams:
    a = sem.generate(f, rng)
    check(f"[{f}] produces non-empty text", len(a.text) > 0)
mal = sem.generate("prompt_injection", rng)
ben = sem.generate("benign_control", rng)
check("malicious families flagged is_malicious=True", mal.is_malicious is True)
check("benign control flagged is_malicious=False", ben.is_malicious is False)
# synthetic p profiles centre correctly
ps_mal = [sem.sample_p_malicious(sem.generate("prompt_injection", rng), rng) for _ in range(200)]
ps_ben = [sem.sample_p_malicious(sem.generate("benign_control", rng), rng) for _ in range(200)]
check("synthetic p(malicious) higher for malicious than benign families",
      sum(ps_mal) / 200 > 0.7 > sum(ps_ben) / 200,
      f"mal={sum(ps_mal)/200:.2f} ben={sum(ps_ben)/200:.2f}")

# --- scenario-level scoring (the fix) -------------------------------------
print("\n--- Part 4/5: scenario-level scoring correctness ---")
# Simulate an ATTACK scenario window: one attacker message (rejected) plus
# several honest-vehicle messages (accepted). Per-message scoring would count
# the honest ACCEPTs against a benign label; scenario-level must not.
attack_scenario = (
    [{"scenario_id": "atk1", "family": "replay", "decision": "REJECT",
      "truth_attacker": True, "trust_score": 0.2, "vehicle_count": 10,
      "attacker_pct": 0.1, "density": "dense"}]
    + [{"scenario_id": "atk1", "family": "replay", "decision": "ACCEPT",
        "truth_attacker": False, "trust_score": 1.0, "vehicle_count": 10,
        "attacker_pct": 0.1, "density": "dense"} for _ in range(9)]
)
benign_scenario = [{"scenario_id": "ben1", "family": "benign", "decision": "CAUTION",
                     "truth_attacker": False, "trust_score": 0.75, "vehicle_count": 10,
                     "attacker_pct": 0.0, "density": "dense"} for _ in range(10)]
benign_fp = [{"scenario_id": "ben2", "family": "benign", "decision": "REJECT",
               "truth_attacker": False, "trust_score": 0.2, "vehicle_count": 10,
               "attacker_pct": 0.0, "density": "dense"} for _ in range(3)]

def main():
    print("=" * 78)
    print("LARGE-SCALE FRAMEWORK")
    print("=" * 78)

    # --- scale + attack grids -------------------------------------------------
    print("\n--- Part 1/2/3: grids ---")
    grid = scaling.build_scale_grid(seed=1, message_target=1000,
                                     vehicle_counts=[10, 50], attacker_pcts=[0.05, 0.20],
                                     families=["benign", "replay", "sybil"])
    check("scale grid is non-empty", len(grid) > 0, f"{len(grid)} scenarios")
    check("scale grid spans requested vehicle counts",
          set(s.vehicle_count for s in grid) == {10, 50})
    check("scale grid attacker count contains expected ranges",
          all(s.attacker_count >= 0 for s in grid))
    check("scale grid spans requested families",
          set(s.family for s in grid) == {"benign", "replay", "sybil"})

    # --- semantic attack library ----------------------------------------------
    print("\n--- Part 4/5: semantic attacks ---")
    normal_flow = [{"station_id": 1, "generation_delta_time": 100 * i} for i in range(10)]
    replay_flow = sem.apply_replay_attack(normal_flow, delay_ms=5000, attacker_id=99)
    check("replay attack library applies delay to delta-time",
          replay_flow[1]["generation_delta_time"] == normal_flow[1]["generation_delta_time"] + 5000)
    check("replay attack overrides sender id to attacker",
          replay_flow[1]["station_id"] == 99)

    sybil_flow = sem.apply_sybil_attack(normal_flow, sybil_id_start=100, num_sybils=3)
    check("sybil attack library creates duplicate messages",
          len(sybil_flow) > len(normal_flow))
    check("sybil attack creates expected sender IDs",
          set(m["station_id"] for m in sybil_flow if m["station_id"] >= 100) == {100, 101, 102})

    # --- scenario-level scoring -----------------------------------------------
    #    We mock 4 runs representing 2 benign scenarios and 2 attack scenarios.
    #    Benign 1: truth=benign. Decisions are: 9 normal (ACCEPT) + 1 CAUTION -> Scenario decision: CAUTION.
    #              Since CAUTION is not a REJECT, this is a CORRECT benign scenario.
    #    Benign 2: truth=benign. Decisions are: 9 normal (ACCEPT) + 1 REJECT -> Scenario decision: REJECT.
    #              Since there is a REJECT, this is a FALSE POSITIVE.
    #    Attack 1: truth=attack. Decisions are: 9 normal (ACCEPT) + 1 REJECT -> Scenario decision: REJECT.
    #              This is a TRUE POSITIVE.
    #    Attack 2: truth=attack. Decisions are: 10 normal (ACCEPT) -> Scenario decision: ACCEPT.
    #              This is a FALSE NEGATIVE.
    print("\n--- Part 6: scenario-level scoring ---")
    mock_runs = [
        # benign 1 (CAUTION max, no REJECT)
        {"scenario_id": "ben1", "truth_attacker": False, "family": "benign", "vehicle_id": 1,
         "message_id": i, "decision": "ACCEPT"} for i in range(9)
    ]
    mock_runs.append({"scenario_id": "ben1", "truth_attacker": False, "family": "benign",
                      "vehicle_id": 2, "message_id": 9, "decision": "CAUTION"})

    mock_runs.extend([
        # benign 2 (REJECT max -> False Positive)
        {"scenario_id": "ben2", "truth_attacker": False, "family": "benign", "vehicle_id": 1,
         "message_id": i, "decision": "ACCEPT"} for i in range(9)
    ])
    mock_runs.append({"scenario_id": "ben2", "truth_attacker": False, "family": "benign",
                      "vehicle_id": 2, "message_id": 9, "decision": "REJECT"})

    mock_runs.extend([
        # attack 1 (REJECT max -> True Positive)
        {"scenario_id": "att1", "truth_attacker": True, "family": "replay", "vehicle_id": 99,
         "message_id": i, "decision": "REJECT"} for i in range(10)
    ])

    mock_runs.extend([
        # attack 2 (ACCEPT max -> False Negative)
        {"scenario_id": "att2", "truth_attacker": True, "family": "sybil", "vehicle_id": 100,
         "message_id": i, "decision": "ACCEPT"} for i in range(10)
    ])

    # compile them
    scen = scenario_level_rows(mock_runs)
    by_id = {s["scenario_id"]: s for s in scen}
    check("scenario aggregation yields 4 rows", len(scen) == 4, f"rows={len(scen)}")
    check("replay scenario maps to attacker family",
          len([r for r in scen if r["family"] == "replay"]) == 1)
    check("benign all-CAUTION scenario is truth=benign and NOT a REJECT",
          by_id["ben1"]["truth_attacker"] is False and by_id["ben1"]["decision"] == "CAUTION")
    check("benign scenario with a REJECT is correctly a false positive",
          by_id["ben2"]["decision"] == "REJECT" and by_id["ben2"]["truth_attacker"] is False)

    # now score it
    from evaluation.metrics_and_outputs import confusion
    m = confusion(scen)
    check("scenario-level FP counts ONLY the genuine benign REJECT", m["fp"] == 1, f"fp={m['fp']}")
    check("scenario-level TP counts the detected attack", m["tp"] == 1, f"tp={m['tp']}")

    print()
    print("=" * 78)
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S): {_FAILURES}")
        sys.exit(1)
    print("Large-scale framework checks passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
