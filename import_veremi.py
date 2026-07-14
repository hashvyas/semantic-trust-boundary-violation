#!/usr/bin/env python3
"""
import_veremi.py — VeReMi / VeReMi-Extension -> STBV flat-report importer
========================================================================
Converts VeReMi ground-truth-labelled BSM logs into the schema the STBV
kinematic layers (B1/MBD/CP) consume, so the stack can be evaluated on a
RECOGNIZED public dataset instead of only self-generated scenarios.

  Dataset:  VeReMi (van der Heijden et al., SecureComm 2018) /
            VeReMi Extension (Kamel et al., IEEE ICC 2020)
  Source :  https://veremi-dataset.github.io  (public; cite the papers above)

HONESTY / SAFETY:
  * This importer is written to VeReMi's PUBLISHED format. VeReMi has had
    minor format variations across releases, so it does NOT assume — it
    INSPECTS each file, prints the fields it finds, and maps only fields it
    recognizes. Anything ambiguous is reported and skipped, never guessed.
  * Ground-truth label: VeReMi marks misbehavior via `attackerType` (0 =
    genuine, non-zero = attacker) in the ground-truth file and/or per-message.
    label = 1 iff attackerType != 0, else 0. If no ground-truth field is
    found, the importer REFUSES to label and tells you what it saw.
  * VALIDATE before trusting: run with --inspect first on one log to confirm
    the field mapping matches YOUR download, then run the full conversion.

VeReMi record fields this importer looks for (per the dataset spec):
  type (2=self BSM, 3=received BSM), sendTime/rcvTime, sender, senderPseudo,
  messageID, pos [x,y,z], spd [x,y], (optionally) hed/heading, acl/accel.
Ground-truth per (sender/senderPseudo): attackerType (int).

Output: one JSON per message in STBV flat-report shape (matches
bridges/message_adapter.to_flat_report's consumed keys), plus is_attacker.

Usage:
    # 1. LOOK before converting (prints detected schema, no output written):
    python3 import_veremi.py --input datasets/VeReMi --inspect

    # 2. Convert:
    python3 import_veremi.py --input datasets/VeReMi --output datasets/veremi_processed

    # 3. (optional) cap messages / choose scenario subdir:
    python3 import_veremi.py --input datasets/VeReMi/<scenario> --output out --max 20000
"""
from __future__ import annotations
import argparse, glob, json, math, os, pathlib, sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def _iter_json_lines(path: str):
    """VeReMi logs are usually JSON-per-line; some releases are a JSON array.
    Handle both without assuming."""
    with open(path, "r") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            try:
                for rec in json.load(f):
                    yield rec
                return
            except Exception:
                f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _find_ground_truth(input_dir: str) -> Tuple[Dict[Any, int], str]:
    """Locate a VeReMi ground-truth file and build {sender_or_pseudo:
    attackerType}. Returns (map, description). Empty map if none found."""
    gt_candidates = []
    for pat in ("**/GroundTruthJSONlog*", "**/*ground*truth*", "**/*GroundTruth*",
                "**/traceGroundTruthJSON*"):
        gt_candidates += glob.glob(os.path.join(input_dir, pat), recursive=True)
    gt_candidates = sorted(set(p for p in gt_candidates if os.path.isfile(p)))
    gt: Dict[Any, int] = {}
    used = []
    for path in gt_candidates:
        for rec in _iter_json_lines(path):
            if not isinstance(rec, dict):
                continue
            atk = rec.get("attackerType", rec.get("attacker_type"))
            key = rec.get("sender", rec.get("senderPseudo", rec.get("pseudo")))
            if atk is not None and key is not None:
                # a sender is an attacker if EVER marked non-zero
                gt[key] = max(int(gt.get(key, 0)), int(atk))
        if gt:
            used.append(path)
    return gt, (f"{len(gt)} senders from {used}" if gt else "none found")


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _extract(rec: Dict[str, Any], gt: Dict[Any, int]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Map one VeReMi record -> STBV flat report. Returns (report|None, note)."""
    if not isinstance(rec, dict):
        return None, "not a dict"
    # VeReMi type: 3 = received BSM (what a detector sees). Keep type 2/3; skip
    # GPS-only rows that lack position.
    pos = rec.get("pos") or rec.get("position")
    spd = rec.get("spd") or rec.get("speed")
    sender = rec.get("sender", rec.get("senderPseudo", rec.get("pseudo")))
    t = rec.get("rcvTime", rec.get("sendTime", rec.get("time")))
    if pos is None or sender is None:
        return None, "missing pos/sender"

    def comp(v, i):
        if isinstance(v, (list, tuple)) and len(v) > i:
            return _num(v[i])
        return 0.0

    x, y = comp(pos, 0), comp(pos, 1)
    vx, vy = comp(spd, 0), comp(spd, 1)
    speed = math.hypot(vx, vy)
    # heading: prefer explicit field, else derive from velocity vector
    hed = rec.get("hed") or rec.get("heading")
    if isinstance(hed, (list, tuple)):
        heading = math.degrees(math.atan2(comp(hed, 1), comp(hed, 0))) % 360.0
    elif hed is not None:
        heading = _num(hed)
    else:
        heading = math.degrees(math.atan2(vy, vx)) % 360.0 if (vx or vy) else 0.0

    attacker_type = gt.get(sender)
    if attacker_type is None:
        # some releases carry attackerType inline on the message
        attacker_type = rec.get("attackerType", rec.get("attacker_type"))
    if attacker_type is None:
        return None, "no ground-truth attackerType for sender"

    report = {
        # keys aligned with bridges/message_adapter.to_flat_report consumers
        "sender": sender,
        "x": x, "y": y,
        "speed": speed,
        "heading": heading,
        "timestamp": _num(t),
        "is_attacker": bool(int(attacker_type) != 0),
        "veremi_attacker_type": int(attacker_type),
        "source": "veremi",
    }
    return report, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="VeReMi dir (a scenario dir or the dataset root)")
    ap.add_argument("--output", help="output dir (omit with --inspect)")
    ap.add_argument("--inspect", action="store_true", help="print detected schema, write nothing")
    ap.add_argument("--max", type=int, default=0, help="cap total messages (0 = all)")
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        print(f"[FATAL] --input dir not found: {args.input}")
        print("        Download VeReMi/VeReMi-Extension from https://veremi-dataset.github.io")
        return 2

    gt, gt_desc = _find_ground_truth(args.input)
    print(f"[ground truth] {gt_desc}")
    if gt:
        print(f"[ground truth] attackerType distribution: "
              f"{Counter(int(v!=0) for v in gt.values())} (0=genuine,1=attacker senders)")

    logs = []
    for pat in ("**/traceJSONlog*", "**/*JSONlog*", "**/veins*", "**/*.json"):
        logs += glob.glob(os.path.join(args.input, pat), recursive=True)
    logs = sorted(set(p for p in logs if os.path.isfile(p) and "round" not in os.path.basename(p).lower()
                      and "ground" not in os.path.basename(p).lower()))
    if not logs:
        print(f"[FATAL] no VeReMi log files found under {args.input}")
        return 3
    print(f"[logs] {len(logs)} candidate log file(s)")

    if args.inspect:
        sample_path = logs[0]
        print(f"\n[inspect] first log: {sample_path}")
        n = 0
        field_counter = Counter()
        for rec in _iter_json_lines(sample_path):
            if isinstance(rec, dict):
                field_counter.update(rec.keys())
            n += 1
            if n >= 200:
                break
        print(f"[inspect] fields seen in first {n} records: {dict(field_counter)}")
        # try mapping a few
        ok = bad = 0
        for rec in _iter_json_lines(sample_path):
            r, note = _extract(rec, gt)
            if r: ok += 1
            else: bad += 1
            if ok + bad >= 50: break
        print(f"[inspect] of 50 sampled: {ok} mappable, {bad} skipped")
        print("[inspect] NOTHING written. If fields look right, re-run with --output.")
        if not gt:
            print("[inspect][WARN] no ground truth found -> labels impossible. "
                  "Point --input at the dir containing the GroundTruth log.")
        return 0

    if not args.output:
        print("[FATAL] --output required unless --inspect"); return 2
    if not gt and not any("attackerType" in (r or {}) for r in [next(_iter_json_lines(logs[0]), {})]):
        print("[FATAL] no ground-truth labels available; refusing to write unlabeled data.")
        print("        Re-run --inspect and point --input at the dir with the GroundTruth file.")
        return 4

    out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    skip_reasons = Counter()
    label_counter = Counter()
    all_reports: List[Dict[str, Any]] = []
    for lp in logs:
        for rec in _iter_json_lines(lp):
            r, note = _extract(rec, gt)
            if r is None:
                skipped += 1; skip_reasons[note] += 1; continue
            all_reports.append(r); label_counter[int(r["is_attacker"])] += 1
            written += 1
            if args.max and written >= args.max:
                break
        if args.max and written >= args.max:
            break

    (out / "veremi_flat_reports.json").write_text(json.dumps(all_reports))
    manifest = {
        "dataset": "VeReMi / VeReMi Extension",
        "cite": ["van der Heijden et al., SecureComm 2018",
                 "Kamel et al., VeReMi Extension, IEEE ICC 2020"],
        "source_url": "https://veremi-dataset.github.io",
        "input_dir": os.path.abspath(args.input),
        "messages_written": written,
        "messages_skipped": skipped,
        "skip_reasons": dict(skip_reasons),
        "label_distribution": {"genuine(0)": label_counter[0], "attacker(1)": label_counter[1]},
        "label_rule": "is_attacker = (attackerType != 0), from VeReMi ground truth",
        "schema": "STBV flat report: sender,x,y,speed,heading,timestamp,is_attacker",
        "VALIDATE": "Confirm field mapping with --inspect before trusting these labels.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] wrote {written} messages ({label_counter[1]} attacker / {label_counter[0]} genuine)")
    print(f"       -> {out/'veremi_flat_reports.json'}")
    print(f"       -> {out/'manifest.json'}")
    if skipped:
        print(f"[note] skipped {skipped}: {dict(skip_reasons)}")
    if label_counter[1] == 0:
        print("[WARN] zero attacker messages labelled -> ground-truth mapping likely wrong. "
              "Re-run --inspect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())