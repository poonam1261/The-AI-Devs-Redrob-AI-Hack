from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def stream_candidates(path: str | Path) -> Iterable[dict[str, Any]]:
    src = Path(path)
    with src.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_candidate_ids(path: str | Path) -> set[str]:
    return {candidate["candidate_id"] for candidate in stream_candidates(path)}


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
