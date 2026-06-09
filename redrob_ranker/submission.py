from __future__ import annotations

import csv
from pathlib import Path

from .explain import build_explanation, make_csv_reason
from .io import write_json
from .jd import JobIntent
from .scoring import RankedCandidate


HEADER = ["candidate_id", "rank", "score", "reasoning"]


def write_submission(path: str | Path, ranked: list[RankedCandidate], intent: JobIntent) -> None:
    out = Path(path)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for item in ranked:
            writer.writerow(
                [
                    item.candidate.candidate_id,
                    item.rank,
                    f"{item.final_score:.6f}",
                    make_csv_reason(item, intent),
                ]
            )


def write_explanations(path: str | Path, ranked: list[RankedCandidate], intent: JobIntent) -> None:
    payload = [build_explanation(item, intent) for item in ranked]
    write_json(path, payload)
