#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extra local checks for a Redrob submission.")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--candidates", required=True)
    return parser.parse_args()


def load_ids(path: Path) -> set[str]:
    import json

    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(json.loads(line)["candidate_id"])
    return ids


def main() -> int:
    args = parse_args()
    submission = Path(args.submission)
    candidate_ids = load_ids(Path(args.candidates))

    errors: list[str] = []
    with submission.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if reader.fieldnames != ["candidate_id", "rank", "score", "reasoning"]:
        errors.append("Header must be candidate_id,rank,score,reasoning.")
    if len(rows) != 100:
        errors.append(f"Expected 100 data rows; found {len(rows)}.")

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    previous_score: float | None = None
    for index, row in enumerate(rows, start=2):
        cid = row.get("candidate_id", "")
        if cid not in candidate_ids:
            errors.append(f"Row {index}: candidate_id {cid} does not exist in candidates file.")
        if cid in seen_ids:
            errors.append(f"Row {index}: duplicate candidate_id {cid}.")
        seen_ids.add(cid)

        try:
            rank = int(row.get("rank", ""))
        except ValueError:
            errors.append(f"Row {index}: rank is not an integer.")
            continue
        if rank in seen_ranks:
            errors.append(f"Row {index}: duplicate rank {rank}.")
        seen_ranks.add(rank)

        try:
            score = float(row.get("score", ""))
        except ValueError:
            errors.append(f"Row {index}: score is not a float.")
            continue
        if previous_score is not None and score > previous_score:
            errors.append(f"Row {index}: score increased from previous row.")
        previous_score = score

        reasoning = row.get("reasoning", "")
        if len(reasoning.strip()) < 40:
            errors.append(f"Row {index}: reasoning is too short to be useful.")

    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"Missing ranks: {sorted(missing)}")

    if errors:
        print(f"Local validation failed ({len(errors)} issue(s)):")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Local validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
