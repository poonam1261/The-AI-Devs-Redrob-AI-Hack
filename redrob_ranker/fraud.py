from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .config import RankerConfig
from .jd import JobIntent
from .normalize import NormalizedCandidate
from .retrieval import count_phrase_hits


NONTECH_TITLES = {
    "marketing manager",
    "sales executive",
    "hr manager",
    "accountant",
    "graphic designer",
    "content writer",
    "operations manager",
    "customer support",
    "civil engineer",
    "mechanical engineer",
    "business analyst",
    "project manager",
}

SERVICES_COMPANIES = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
    "hcl",
    "tech mahindra",
    "mphasis",
}


@dataclass(frozen=True)
class FraudAssessment:
    penalty: float
    flags: tuple[str, ...]
    consistency_score: float


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _months_between(start: dt.date, end: dt.date) -> int:
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))


def assess_fraud(
    candidate: NormalizedCandidate,
    intent: JobIntent,
    config: RankerConfig,
    today: dt.date = dt.date(2026, 6, 9),
) -> FraudAssessment:
    flags: list[str] = []
    penalty = 0.0

    expert_zero = [
        skill
        for skill in candidate.skills
        if skill.get("proficiency") == "expert" and int(skill.get("duration_months") or 0) <= 1
    ]
    advanced_zero = [
        skill
        for skill in candidate.skills
        if skill.get("proficiency") in {"advanced", "expert"}
        and int(skill.get("duration_months") or 0) <= 1
    ]
    expert_count = sum(1 for skill in candidate.skills if skill.get("proficiency") == "expert")
    if expert_zero:
        penalty += min(0.24, len(expert_zero) * config.fraud["expert_zero_duration_penalty"])
        flags.append(f"{len(expert_zero)} expert skills have near-zero duration")
    if len(advanced_zero) >= 3:
        penalty += 0.10
        flags.append("multiple advanced/expert skills have near-zero duration")
    if expert_count >= 10:
        penalty += config.fraud["skill_inflation_penalty"]
        flags.append("unusually broad expert skill list")

    full_hits = count_phrase_hits(candidate.full_text, intent.domain_terms + intent.vector_terms)
    career_hits = count_phrase_hits(candidate.career_text, intent.domain_terms + intent.vector_terms)
    title_low = candidate.title.lower()
    if title_low in NONTECH_TITLES and full_hits >= 5 and career_hits <= 1:
        penalty += config.fraud["nontechnical_ai_penalty"]
        flags.append("nontechnical title claims AI/search depth without career evidence")
    if full_hits >= 7 and career_hits <= 1:
        penalty += config.fraud["unsupported_ai_claim_penalty"]
        flags.append("AI/search skill claims are weakly supported by career history")

    timeline_issue = False
    total_months = 0
    for item in candidate.career:
        start = _parse_date(item.get("start_date"))
        end = _parse_date(item.get("end_date")) or today
        if start and start > today:
            timeline_issue = True
        if end and end > today:
            timeline_issue = True
        if start and end and end < start:
            timeline_issue = True
        if start and end:
            calculated = _months_between(start, end)
            total_months += calculated
            stated = int(item.get("duration_months") or calculated)
            if abs(stated - calculated) > 6:
                timeline_issue = True
    if timeline_issue:
        penalty += config.fraud["timeline_penalty"]
        flags.append("employment timeline or duration is inconsistent")
    if total_months and abs((total_months / 12.0) - candidate.years) > 4.0:
        penalty += 0.08
        flags.append("profile experience differs materially from employment history")

    career_companies = {str(item.get("company", "")).strip().lower() for item in candidate.career}
    if career_companies and career_companies.issubset(SERVICES_COMPANIES):
        penalty += config.fraud["services_only_penalty"]
        flags.append("career appears services-only")

    penalty = min(config.fraud["max_penalty"], penalty)
    return FraudAssessment(
        penalty=penalty,
        flags=tuple(flags),
        consistency_score=max(0.0, 1.0 - penalty),
    )
