from __future__ import annotations

from dataclasses import dataclass

from .config import RankerConfig
from .features import FeatureBreakdown
from .fraud import FraudAssessment
from .normalize import NormalizedCandidate


@dataclass(frozen=True)
class RankedCandidate:
    candidate: NormalizedCandidate
    features: FeatureBreakdown
    fraud: FraudAssessment
    base_score: float
    final_score: float
    rank: int = 0

    @property
    def heap_key(self) -> tuple[float, float, float, float, int]:
        return (
            self.final_score,
            self.features.semantic_fit,
            self.features.domain_fit,
            self.features.production_experience,
            -_id_num(self.candidate.candidate_id),
        )


def _id_num(candidate_id: str) -> int:
    try:
        return int(candidate_id.split("_", 1)[1])
    except Exception:
        return 0


def score_candidate(
    candidate: NormalizedCandidate,
    features: FeatureBreakdown,
    fraud: FraudAssessment,
    config: RankerConfig,
) -> RankedCandidate:
    weights = config.weights
    base = (
        weights["semantic_fit"] * features.semantic_fit
        + weights["domain_fit"] * features.domain_fit
        + weights["production_experience"] * features.production_experience
        + weights["behavioral_score"] * features.behavioral_score
        + weights["career_quality"] * features.career_quality
    )
    final = max(0.0, min(1.0, base * (1.0 - fraud.penalty)))
    return RankedCandidate(
        candidate=candidate,
        features=features,
        fraud=fraud,
        base_score=base,
        final_score=final,
    )


def sort_ranked(items: list[RankedCandidate]) -> list[RankedCandidate]:
    ordered = sorted(
        items,
        key=lambda item: (
            -item.final_score,
            -item.features.semantic_fit,
            -item.features.domain_fit,
            -item.features.production_experience,
            item.candidate.candidate_id,
        ),
    )
    return [
        RankedCandidate(
            candidate=item.candidate,
            features=item.features,
            fraud=item.fraud,
            base_score=item.base_score,
            final_score=item.final_score,
            rank=index,
        )
        for index, item in enumerate(ordered, start=1)
    ]
