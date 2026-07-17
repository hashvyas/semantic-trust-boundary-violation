#!/usr/bin/env python3
"""
tools/convert_dataset_timestamps.py
===================================
Script to recursively convert absolute timestamps to scenario-relative
integer milliseconds in test_messages and scenarios datasets.
"""
from __future__ import annotations
import os
import json
import pathlib
import sys

TIMESTAMP_KEYS = ["timestamp", "generation_delta_time"]


def process_message(msg: dict, first_ts: int | None = None) -> tuple[dict, int | None]:
    """Process a message and return the updated dict and the first found timestamp (if not set)."""
    ts_found = None

    # 1. Look in root level
    for k in TIMESTAMP_KEYS:
        if k in msg and isinstance(msg[k], (int, float)):
            ts_found = int(msg[k])
            break

    # 2. Look in msg["cam"] if present
    if ts_found is None and "cam" in msg and isinstance(msg["cam"], dict):
        for k in TIMESTAMP_KEYS:
            if k in msg["cam"] and isinstance(msg["cam"][k], (int, float)):
                ts_found = int(msg["cam"][k])
                break

    if ts_found is None:
        return msg, first_ts

    if first_ts is None:
        first_ts = ts_found

    # Rebase timestamp
    rebased_ts = ts_found - first_ts

    # Write back
    for k in TIMESTAMP_KEYS:
        if k in msg and isinstance(msg[k], (int, float)):
            msg[k] = rebased_ts

    if "cam" in msg and isinstance(msg["cam"], dict):
        for k in TIMESTAMP_KEYS:
            if k in msg["cam"] and isinstance(msg["cam"][k], (int, float)):
                msg["cam"][k] = rebased_ts

    return msg, first_ts


def convert_directory(dir_path: pathlib.Path):
    json_files = sorted(list(dir_path.glob("*.json")))
    if not json_files:
        return

    # Find the minimum timestamp across all files in this directory
    all_ts = []

    def extract_ts(msg: Any) -> int | None:
        if not isinstance(msg, dict):
            return None
        for k in TIMESTAMP_KEYS:
            if k in msg and isinstance(msg[k], (int, float)):
                return int(msg[k])
        if "cam" in msg and isinstance(msg["cam"], dict):
            for k in TIMESTAMP_KEYS:
                if k in msg["cam"] and isinstance(msg["cam"][k], (int, float)):
                    return int(msg["cam"][k])
        return None

    for p in json_files:
        if p.name == "metadata.json":
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if isinstance(data, dict):
            ts = extract_ts(data)
            if ts is not None:
                all_ts.append(ts)
        elif isinstance(data, list):
            for msg in data:
                ts = extract_ts(msg)
                if ts is not None:
                    all_ts.append(ts)

    if not all_ts:
        return

    min_ts = min(all_ts)
    print(f"Rebasing directory {dir_path} with min_ts={min_ts}")
    for p in json_files:
        if p.name == "metadata.json":
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if isinstance(data, dict):
            data, _ = process_message(data, min_ts)
            with p.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
        elif isinstance(data, list):
            new_data = []
            for msg in data:
                if isinstance(msg, dict):
                    msg, _ = process_message(msg, min_ts)
                new_data.append(msg)
            with p.open("w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=2)
                f.write("\n")


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    targets = [
        root / "test_messages",
        root / "scenarios"
    ]
    for target in targets:
        if not target.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(target):
            convert_directory(pathlib.Path(dirpath))


if __name__ == "__main__":
    main()
