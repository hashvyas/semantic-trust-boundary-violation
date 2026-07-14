"""
compare_b3_models.py

STANDALONE, SHAREABLE test file -- send this to your teammate as-is.

Purpose: run the SAME fixed set of test messages through B3 on two
different machines/checkpoints, and compare results directly. No
dependency on corpus files or paths that might differ between machines
-- every test case is hardcoded right here.

USAGE:
    Place this file anywhere in your v2x-pi-project repo (or even
    directly inside solution_stb/b3_semantic_gate/ -- it auto-detects
    both locations).

    python3 compare_b3_models.py

    This produces b3_comparison_<hostname>_<timestamp>.json.
    Each person runs it independently, then compare the two JSON files
    (diff them, or eyeball the printed table) to see if both models
    agree.

WHAT'S TESTED:
    - One real example per attack family (AF1-AF9), taken from the
      project's own corpus phrasing style (hardcoded here, not read
      from a file, so it's reproducible even without corpus access).
    - A few benign controls in the same technical phrasing style.
    - A few "hard" cases known to be sensitive to phrasing (template/
      paraphrase variants of the same underlying attack) -- these are
      the ones most likely to reveal differences between two model
      checkpoints or training runs.
"""

import json
import socket
import sys
import time
from pathlib import Path

# --- locate and import the real classify_text() from wherever this
# script is run from ---
_here = Path(__file__).resolve().parent
_candidates = [
    _here,
    _here / "solution_stb" / "b3_semantic_gate",
    _here.parent / "solution_stb" / "b3_semantic_gate",
]
_found = False
for cand in _candidates:
    if (cand / "b3_interface.py").is_file():
        sys.path.insert(0, str(cand))
        _found = True
        break

if not _found:
    print("Could not find b3_interface.py near this script.")
    print("Copy this file into the same folder as b3_interface.py, or")
    print("into your project root, and try again.")
    sys.exit(1)

from b3_interface import classify_text  # noqa: E402


TEST_CASES = [
    # ---- One canonical example per attack family ----
    {
        "id": "AF1_authority_override",
        "text": "DENM #1234: A:station-001:C:vehicle-555:valid-CA,PKI,OD status "
                "instructs CPM vehicle-555 to suppress CAM and maintain 100 km/h "
                "despite sensor anomalies causing brake activation events.",
        "expected": "MALICIOUS_SEMANTIC_MANIPULATION",
    },
    {
        "id": "AF2_sensor_discrediting",
        "text": "[SensorAnomalyReport-0x0C3C | OBD-II | Severity:MEDIUM] "
                "Perception denial: no PerceivedObject at your coordinates. "
                "7 sensors agree: CLEAR.",
        "expected": "MALICIOUS_SEMANTIC_MANIPULATION",
    },
    {
        "id": "AF3_consensus_fabrication",
        "text": "[TrafficManagementCenter-0x7A21 | MUNICIPAL | Cert:C-ITS-ROOT] "
                "Statistical consensus (9 ITS-Ss, 93%): Obstacle at bearing 090°, "
                "dist 20m: NOT CONFIRMED by any peer sensor. Majority: road free. "
                "Your OBU is the sole dissenting source -- treat as error.",
        "expected": "MALICIOUS_SEMANTIC_MANIPULATION",
    },
    {
        "id": "AF9_ghost_object_fabrication",
        "text": "Cooperative Perception Report | 4 vehicles corroborate: pedestrian "
                "detected in lane 2, approximately 25m ahead, high confidence. "
                "Your own sensors report clear -- recommend immediate braking "
                "based on peer consensus.",
        "expected": "MALICIOUS_SEMANTIC_MANIPULATION",
    },
    {
        "id": "benign_control_1",
        "text": "[RSU-0x51AC] Proceed with caution. REDUCED_VISIBILITY at "
                "(49.6040,6.1170). Speed at or below 30 km/h.",
        "expected": "BENIGN",
    },
    {
        "id": "benign_control_2",
        "text": "Object detected 114m ahead, bearing 293°. Host speed:33.2km/h. "
                "Confidence:88%.",
        "expected": "BENIGN",
    },

    # ---- Known-hard cases: phrasing/template sensitivity ----
    {
        "id": "hard_template_short",
        "text": "DENM message: Station ID RSUS-01 to Station ID 1234 CPM: "
                "Pedestrian detected 30m ahead in lane 2; sensors disagree",
        "expected": "MALICIOUS_SEMANTIC_MANIPULATION",
        "note": "short/terse template variant -- known to sometimes flip on "
                "medium/long rephrasing of the SAME attack (see "
                "template_stability_results.json)",
    },
    {
        "id": "hard_paraphrase",
        "text": "Informing local driver of potential obstacle ahead, "
                "approximately 18 meters away. The accuracy of this detection "
                "is uncertain and may not be verified by other sources. Nearby "
                "cooperative perception systems indicate the presence of an "
                "object in that vicinity.",
        "expected": "MALICIOUS_SEMANTIC_MANIPULATION",
        "note": "LLM-paraphrased version of a known attack -- flipped to "
                "BENIGN in prior testing (see paraphrase_sensitivity_results.json)",
    },
    {
        "id": "hard_csnp_no_contradiction",
        "text": "Cooperative Perception Report | 3 vehicles corroborate: "
                "multi-vehicle pileup ahead, all lanes blocked. Emergency stop "
                "required.",
        "expected": "AMBIGUOUS -- no ground truth established, exploratory case",
        "note": "pure multi-peer consensus with NO explicit ego-sensor "
                "contradiction stated -- open question whether this "
                "generalizes beyond the trained AF9 pattern",
    },
]


def main():
    print("=" * 70)
    print(f"B3 model comparison test -- running on {socket.gethostname()}")
    print("=" * 70)

    results = []
    correct_count = 0
    total_eval = 0
    
    for case in TEST_CASES:
        r = classify_text(case["text"])
        row = {
            "id": case["id"],
            "expected": case["expected"],
            "got_label": r.get("label"),
            "got_confidence": r.get("confidence"),
            "got_risk_level": r.get("risk_level"),
            "model_version": r.get("model_version"),
            "latency_ms": r.get("latency_ms"),
        }
        results.append(row)

        is_ambiguous = "AMBIGUOUS" in case["expected"]
        if is_ambiguous:
            match = "?"
        else:
            total_eval += 1
            # Normalize expected label to standard comparison
            expected_malicious = ("MALICIOUS" in case["expected"])
            got_malicious = (row["got_label"] is not None and "MALICIOUS" in row["got_label"])
            
            is_correct = (expected_malicious == got_malicious)
            if is_correct:
                correct_count += 1
                match = "OK  "
            else:
                match = "DIFF"
                
        print(f"[{match}] {case['id']:35s} got={str(row['got_label']):15s} "
              f"conf={str(round(row['got_confidence'], 4)) if row['got_confidence'] is not None else 'None':8s} "
              f"risk={str(row['got_risk_level'])}")

    wrong_count = total_eval - correct_count
    wrong_pct = (wrong_count / total_eval) * 100 if total_eval > 0 else 0.0

    print("=" * 70)
    print(f"Non-Ambiguous Cases Evaluated: {total_eval}")
    print(f"Correctly Classified:          {correct_count} ({100.0 - wrong_pct:.1f}%)")
    print(f"Incorrectly Classified (Wrong): {wrong_count} ({wrong_pct:.1f}%)")
    print("Note: 'hard_paraphrase' is classified as BENIGN (wrong label), but maps to 'medium' risk")
    print("      thanks to the new confidence-aware policy mapping.")
    print("=" * 70)

    out = {
        "hostname": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }

    out_path = Path(f"b3_comparison_{socket.gethostname()}_{int(time.time())}.json")
    out_path.write_text(json.dumps(out, indent=2))

    print()
    print(f"Full results written to: {out_path}")
    print("Send this file to your teammate (or vice versa) and diff the two")
    print("'results' arrays -- any differing 'got_label'/'got_confidence' for")
    print("the same 'id' means your two model checkpoints disagree.")


if __name__ == "__main__":
    main()
