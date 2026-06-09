from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .jd import JobIntent
from .normalize import NormalizedCandidate


def count_phrase_hits(text: str, phrases: Iterable[str]) -> int:
    low = text.lower()
    # The dataset is large enough that compiling hundreds of regexes per
    # candidate dominates runtime. Substring matching is intentionally used for
    # speed; downstream title/career weighting keeps it from becoming a pure
    # keyword counter.
    return sum(1 for phrase in phrases if phrase.lower() in low)


def clipped_ratio(hits: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return min(1.0, hits / denominator)


def lexical_semantic_score(candidate: NormalizedCandidate, intent: JobIntent) -> float:
    title_text = f"{candidate.title} {candidate.headline}".lower()
    career_text = candidate.career_text.lower()
    full_text = candidate.full_text.lower()

    target_title = count_phrase_hits(title_text, intent.target_titles)
    adjacent_title = count_phrase_hits(title_text, intent.adjacent_titles)
    title_score = min(1.0, target_title * 0.75 + adjacent_title * 0.25)

    domain_hits = count_phrase_hits(full_text, intent.domain_terms)
    career_domain_hits = count_phrase_hits(career_text, intent.domain_terms)
    vector_hits = count_phrase_hits(full_text, intent.vector_terms)
    production_hits = count_phrase_hits(career_text, intent.production_terms)
    eval_hits = count_phrase_hits(career_text, intent.evaluation_terms)

    evidence_score = (
        0.36 * clipped_ratio(domain_hits, 7)
        + 0.24 * clipped_ratio(career_domain_hits, 4)
        + 0.16 * clipped_ratio(vector_hits, 3)
        + 0.14 * clipped_ratio(production_hits, 4)
        + 0.10 * clipped_ratio(eval_hits, 2)
    )
    return min(1.0, 0.42 * title_score + 0.58 * evidence_score)


@dataclass
class LocalEmbeddingAdapter:
    """Optional local-only embedding adapter.

    The constructor intentionally accepts only filesystem paths. If the model is
    not already present, sentence-transformers will fail instead of downloading.
    """

    model_path: Path

    def __post_init__(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Embedding model path does not exist: {self.model_path}")
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(str(self.model_path), local_files_only=True)

    def cosine_scores(self, job_text: str, texts: list[str], batch_size: int = 128) -> list[float]:
        import numpy as np

        job_vec = self.model.encode([job_text], normalize_embeddings=True)[0]
        scores: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vecs = self.model.encode(batch, normalize_embeddings=True)
            sims = np.dot(vecs, job_vec)
            scores.extend(float((sim + 1.0) / 2.0) for sim in sims)
        return scores


def squash(value: float, sharpness: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-sharpness * value))
