from __future__ import annotations

import argparse
import heapq
import sys
import time
from pathlib import Path

from .config import load_config
from .features import compute_features
from .fraud import assess_fraud
from .io import stream_candidates
from .jd import build_job_intent
from .normalize import normalize_candidate
from .scoring import RankedCandidate, score_candidate, sort_ranked
from .submission import write_explanations, write_submission


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates for a job description.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--job", required=True, help="Path to job_description.docx")
    parser.add_argument("--out", required=True, help="Output submission CSV path")
    parser.add_argument("--config", default="config/weights.yaml", help="Weights YAML path")
    parser.add_argument("--limit", type=int, default=None, help="Shortlist size; defaults to config runtime.shortlist_size")
    parser.add_argument("--max-candidates", type=int, default=None, help="Development-only cap on scanned candidates")
    parser.add_argument("--explanations", default=None, help="Optional explanations JSON output path")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional local sentence-transformers model path. The default ranker does not require it.",
    )
    return parser.parse_args(argv)


def _push_top(heap: list[tuple[tuple[float, float, float, float, int], int, RankedCandidate]], item: RankedCandidate, limit: int, counter: int) -> None:
    entry = (item.heap_key, counter, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.perf_counter()
    config = load_config(args.config)
    limit = int(args.limit or config.runtime.get("shortlist_size", 100))
    intent = build_job_intent(args.job)

    if args.embedding_model:
        model_path = Path(args.embedding_model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Embedding model path {model_path} is not available locally. "
                "Defaulting to hosted downloads is intentionally unsupported."
            )
        print(
            "Note: --embedding-model was provided, but this implementation uses the "
            "deterministic hybrid ranker for the official submission path.",
            file=sys.stderr,
        )

    heap: list[tuple[tuple[float, float, float, float, int], int, RankedCandidate]] = []
    seen = 0
    for seen, raw in enumerate(stream_candidates(args.candidates), start=1):
        if args.max_candidates is not None and seen > args.max_candidates:
            seen -= 1
            break
        candidate = normalize_candidate(raw)
        features = compute_features(candidate, intent)
        fraud = assess_fraud(candidate, intent, config)
        ranked = score_candidate(candidate, features, fraud, config)
        _push_top(heap, ranked, limit, seen)
        if seen % 25000 == 0:
            print(f"processed {seen:,} candidates", file=sys.stderr)

    ranked = sort_ranked([entry[2] for entry in heap])
    write_submission(args.out, ranked, intent)

    explanations_path = args.explanations
    if explanations_path is None:
        out_path = Path(args.out)
        explanations_path = str(out_path.with_name(str(config.runtime.get("explanation_json", "explanations.json"))))
    write_explanations(explanations_path, ranked, intent)

    elapsed = time.perf_counter() - start
    suspicious = sum(1 for item in ranked if item.fraud.penalty >= 0.18)
    print(
        f"ranked {seen:,} candidates; wrote {len(ranked)} rows to {args.out}; "
        f"suspicious_top100={suspicious}; elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
