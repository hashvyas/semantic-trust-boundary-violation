#!/usr/bin/env python3
"""
import_veremi_extension.py — VeReMi Extension flat-report importer
========================================================================
Parses VeReMi Extension JSON logs and outputs deduplicated flat reports in NDJSON format.
"""

from __future__ import annotations
import argparse
import glob
import json
import math
import os
import pathlib
import sys
from collections import Counter
from typing import Any, Dict, List, Set, Tuple


def _iter_json_lines(path: str):
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


def _find_ground_truth(input_dir: str) -> Dict[int, int]:
    """Builds {sender_id: attackerType} mapping from trace log filenames."""
    gt: Dict[int, int] = {}
    
    # 1. Parse attacker types from trace filenames
    log_patterns = ("**/traceJSONlog*", "**/*JSONlog*", "**/veins*", "**/*.json")
    logs = []
    for pat in log_patterns:
        logs += glob.glob(os.path.join(input_dir, pat), recursive=True)
    logs = sorted(set(p for p in logs if os.path.isfile(p) and "round" not in os.path.basename(p).lower()
                      and "ground" not in os.path.basename(p).lower()))
    
    for lp in logs:
        basename = os.path.basename(lp)
        parts = basename.split("-")
        atk_val = None
        atk_idx = -1
        for idx, part in enumerate(parts):
            if part.startswith("A") and part[1:].isdigit():
                atk_val = int(part[1:])
                atk_idx = idx
                break
        if atk_val is not None and atk_idx > 1:
            for i in range(1, atk_idx):
                if parts[i].isdigit():
                    gt[int(parts[i])] = atk_val

    # 2. Check for ground truth logs to augment the mapping
    gt_patterns = ("**/GroundTruthJSONlog*", "**/*ground*truth*", "**/*GroundTruth*",
                   "**/traceGroundTruthJSON*")
    gt_logs = []
    for pat in gt_patterns:
        gt_logs += glob.glob(os.path.join(input_dir, pat), recursive=True)
    gt_logs = sorted(set(p for p in gt_logs if os.path.isfile(p)))
    
    for path in gt_logs:
        for rec in _iter_json_lines(path):
            if not isinstance(rec, dict):
                continue
            atk = rec.get("attackerType", rec.get("attacker_type"))
            sender = rec.get("sender", rec.get("senderPseudo", rec.get("pseudo")))
            if atk is not None and sender is not None:
                gt[int(sender)] = max(gt.get(int(sender), 0), int(atk))

    return gt


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _extract_report(rec: Dict[str, Any], gt: Dict[int, int]) -> Optional[Dict[str, Any]]:
    # type 3 is received BSM
    if rec.get("type") != 3:
        return None
        
    pos = rec.get("pos") or rec.get("position")
    spd = rec.get("spd") or rec.get("speed")
    sender = rec.get("sender")
    t = rec.get("rcvTime", rec.get("sendTime", rec.get("time")))
    message_id = rec.get("messageID")
    
    if pos is None or sender is None or t is None:
        return None
        
    sender_id = int(sender)
    
    def comp(v, i):
        if isinstance(v, (list, tuple)) and len(v) > i:
            return _num(v[i])
        return 0.0
        
    x, y = comp(pos, 0), comp(pos, 1)
    vx, vy = comp(spd, 0), comp(spd, 1)
    speed = math.hypot(vx, vy)
    
    # heading: prefer hed, else derive from speed
    hed = rec.get("hed") or rec.get("heading")
    if isinstance(hed, (list, tuple)):
        heading = math.degrees(math.atan2(comp(hed, 1), comp(hed, 0))) % 360.0
    elif hed is not None:
        heading = _num(hed)
    else:
        heading = math.degrees(math.atan2(vy, vx)) % 360.0 if (vx or vy) else 0.0
        
    attacker_type = gt.get(sender_id, 0)
    
    return {
        "sender": sender_id,
        "x": x,
        "y": y,
        "speed": speed,
        "heading": heading,
        "timestamp": _num(t),
        "messageID": int(message_id) if message_id is not None else None,
        "is_attacker": bool(attacker_type != 0),
        "veremi_attacker_type": attacker_type,
        "source": "veremi",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Importer for VeReMi Extension dataset logs.")
    ap.add_argument("--input_dir", required=True, help="Directory containing the VeReMi Extension log files")
    ap.add_argument("--output", required=True, help="Output filepath for NDJSON flat reports")
    args = ap.parse_args()
    
    input_path = pathlib.Path(args.input_dir)
    if not input_path.is_dir():
        print(f"[FATAL] Input directory '{args.input_dir}' does not exist.")
        return 1
        
    print("[1/3] Building ground truth mapping...")
    gt = _find_ground_truth(args.input_dir)
    print(f"      Mapped {len(gt)} unique vehicles.")
    
    # Locate received log files
    log_files = []
    for pat in ("**/traceJSONlog*", "**/*JSONlog*", "**/veins*", "**/*.json"):
        log_files += glob.glob(os.path.join(args.input_dir, pat), recursive=True)
    log_files = sorted(set(p for p in log_files if os.path.isfile(p) and "ground" not in os.path.basename(p).lower()))
    
    print(f"[2/3] Parsing logs and extracting received reports from {len(log_files)} files...")
    seen_keys: Set[Tuple[int, Any]] = set()
    reports = []
    
    for filepath in log_files:
        for rec in _iter_json_lines(filepath):
            if not isinstance(rec, dict):
                continue
            rep = _extract_report(rec, gt)
            if rep is not None:
                # Deduplication key
                sender = rep["sender"]
                msg_id = rep["messageID"]
                t = rep["timestamp"]
                key = (sender, msg_id) if msg_id is not None else (sender, t)
                
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                reports.append(rep)
                
    # Sort reports chronologically by timestamp
    reports.sort(key=lambda r: r["timestamp"])
    
    print(f"[3/3] Writing {len(reports)} deduplicated flat reports to {args.output}...")
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for r in reports:
            f.write(json.dumps(r) + "\n")
            
    # Compute summary statistics
    unique_senders = {r["sender"] for r in reports}
    benign_senders = {s for s in unique_senders if gt.get(s, 0) == 0}
    attacker_senders = {s for s in unique_senders if gt.get(s, 0) != 0}
    
    print("-" * 60)
    print("VeReMi Extension Import Summary")
    print("-" * 60)
    print(f"Total messages:     {len(reports)}")
    print(f"Unique senders:     {len(unique_senders)}")
    print(f"Benign vehicles:    {len(benign_senders)}")
    print(f"Attacker vehicles:   {len(attacker_senders)}")
    print("-" * 60)
    print("Done successfully.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
