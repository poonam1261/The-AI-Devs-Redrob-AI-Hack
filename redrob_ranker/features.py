from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .fraud import SERVICES_COMPANIES
from .jd import JobIntent
from .normalize import NormalizedCandidate
from .retrieval import clipped_ratio, count_phrase_hits, lexical_semantic_score


PRODUCT_INDUSTRIES = {
    "software",
    "saas",
    "ai/ml",
    "fintech",
    "e-commerce",
    "edtech",
    "food delivery",
    "adtech",
    "transportation",
    "insurance tech",
    "gaming",
    "healthtech",
    "healthtech ai",
    "conversational ai",
}

PRODUCT_COMPANIES = {
    "swiggy",
    "zomato",
    "flipkart",
    "cred",
    "razorpay",
    "meesho",
    "paytm",
    "ola",
    "nykaa",
    "zoho",
    "inmobi",
    "unacademy",
    "vedantu",
    "freshworks",
    "postman",
    "browserstack",
}

RELEVANT_DEGREES = ("computer", "data", "ai", "machine learning", "information", "software", "engineering")
SENIOR_WORDS = ("senior", "staff", "lead", "principal", "architect", "manager", "mentor", "owned", "drove")


@dataclass(frozen=True)
class FeatureBreakdown:
    semantic_fit: float
    domain_fit: float
    production_experience: float
    behavioral_score: float
    career_quality: float
    education_score: float
    title_score: float
    years_score: float
    domain_hits: int
    vector_hits: int
    production_hits: int
    evaluation_hits: int
    product_exposure: float
    services_exposure: float

    def as_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return count_phrase_hits(text, phrases) > 0


def _score_title(candidate: NormalizedCandidate, intent: JobIntent) -> float:
    title_text = f"{candidate.title} {candidate.headline}".lower()
    if _has_any(title_text, intent.target_titles):
        return 1.0
    if _has_any(title_text, intent.adjacent_titles):
        return 0.55
    if _has_any(title_text, intent.negative_titles):
        return 0.04
    if "engineer" in title_text or "scientist" in title_text:
        return 0.35
    return 0.12


def _score_years(years: float, intent: JobIntent) -> float:
    if intent.preferred_experience_min <= years <= intent.preferred_experience_max:
        return 1.0
    if 4.0 <= years < intent.preferred_experience_min or intent.preferred_experience_max < years <= 11.0:
        return 0.72
    if 3.0 <= years <= 13.0:
        return 0.45
    return 0.18


def _product_services_exposure(candidate: NormalizedCandidate) -> tuple[float, float]:
    product = 0
    services = 0
    total = max(1, len(candidate.career))
    for item in candidate.career:
        company = str(item.get("company", "")).lower()
        industry = str(item.get("industry", "")).lower()
        if company in PRODUCT_COMPANIES or industry in PRODUCT_INDUSTRIES:
            product += 1
        if company in SERVICES_COMPANIES or "services" in industry or "consulting" in industry:
            services += 1
    return product / total, services / total


def _score_behavior(candidate: NormalizedCandidate, intent: JobIntent, today: dt.date) -> float:
    signals = candidate.signals
    response = float(signals.get("recruiter_response_rate") or 0.0)
    interview = float(signals.get("interview_completion_rate") or 0.0)
    offer = float(signals.get("offer_acceptance_rate") if signals.get("offer_acceptance_rate") != -1 else 0.45)
    notice = int(signals.get("notice_period_days") or 180)
    response_time = float(signals.get("avg_response_time_hours") or 240.0)
    completeness = float(signals.get("profile_completeness_score") or 0.0) / 100.0
    github = float(signals.get("github_activity_score") if signals.get("github_activity_score") != -1 else 25.0) / 100.0

    last_active_raw = str(signals.get("last_active_date") or "")
    try:
        last_active = dt.date.fromisoformat(last_active_raw)
        days_inactive = max(0, (today - last_active).days)
    except ValueError:
        days_inactive = 365
    recency = 1.0 if days_inactive <= 30 else 0.75 if days_inactive <= 90 else 0.35 if days_inactive <= 180 else 0.08

    notice_score = 1.0 if notice <= 30 else 0.75 if notice <= 60 else 0.42 if notice <= 90 else 0.18
    response_time_score = 1.0 if response_time <= 24 else 0.75 if response_time <= 72 else 0.42 if response_time <= 168 else 0.18
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.35
    verified = (
        float(bool(signals.get("verified_email")))
        + float(bool(signals.get("verified_phone")))
        + float(bool(signals.get("linkedin_connected")))
    ) / 3.0
    location_text = f"{candidate.location} {candidate.country}".lower()
    location_fit = 1.0 if any(loc in location_text for loc in intent.preferred_locations) else 0.65
    if candidate.country.lower() not in intent.preferred_countries:
        location_fit = 0.25
    relocation = 1.0 if signals.get("willing_to_relocate") else 0.55
    work_mode = 1.0 if str(signals.get("preferred_work_mode", "")).lower() in {"hybrid", "flexible"} else 0.72

    return min(
        1.0,
        0.17 * response
        + 0.13 * interview
        + 0.09 * offer
        + 0.14 * recency
        + 0.12 * notice_score
        + 0.08 * response_time_score
        + 0.10 * open_to_work
        + 0.06 * completeness
        + 0.04 * github
        + 0.04 * verified
        + 0.03 * max(location_fit, relocation * 0.8, work_mode * 0.7),
    )


def _score_education(candidate: NormalizedCandidate) -> float:
    if not candidate.education:
        return 0.35
    best = 0.0
    tier_score = {"tier_1": 1.0, "tier_2": 0.78, "tier_3": 0.48, "tier_4": 0.28, "unknown": 0.35}
    for item in candidate.education:
        tier = tier_score.get(str(item.get("tier", "unknown")), 0.35)
        text = f"{item.get('degree', '')} {item.get('field_of_study', '')}".lower()
        relevance = 1.0 if any(term in text for term in RELEVANT_DEGREES) else 0.45
        best = max(best, 0.65 * tier + 0.35 * relevance)
    return best


def compute_features(
    candidate: NormalizedCandidate,
    intent: JobIntent,
    today: dt.date = dt.date(2026, 6, 9),
) -> FeatureBreakdown:
    full_text = candidate.full_text.lower()
    career_text = candidate.career_text.lower()
    semantic = lexical_semantic_score(candidate, intent)
    title = _score_title(candidate, intent)
    years = _score_years(candidate.years, intent)
    product_exposure, services_exposure = _product_services_exposure(candidate)

    domain_hits = count_phrase_hits(full_text, intent.domain_terms)
    career_domain_hits = count_phrase_hits(career_text, intent.domain_terms)
    vector_hits = count_phrase_hits(full_text, intent.vector_terms)
    production_hits = count_phrase_hits(career_text, intent.production_terms)
    evaluation_hits = count_phrase_hits(career_text, intent.evaluation_terms)

    domain = min(
        1.0,
        0.42 * clipped_ratio(career_domain_hits, 4)
        + 0.22 * clipped_ratio(domain_hits, 7)
        + 0.21 * clipped_ratio(vector_hits, 3)
        + 0.15 * title,
    )
    production = min(
        1.0,
        0.38 * clipped_ratio(production_hits, 5)
        + 0.26 * clipped_ratio(evaluation_hits, 2)
        + 0.18 * product_exposure
        + 0.10 * clipped_ratio(count_phrase_hits(career_text, ("owned", "designed", "built", "shipped", "led")), 3)
        + 0.08 * clipped_ratio(count_phrase_hits(career_text, ("10m", "50m", "scale", "latency", "users", "queries")), 2),
    )
    education = _score_education(candidate)
    seniority = clipped_ratio(count_phrase_hits(f"{candidate.title} {candidate.career_text}", SENIOR_WORDS), 3)
    career_quality = min(
        1.0,
        0.34 * years
        + 0.24 * product_exposure
        + 0.16 * seniority
        + 0.12 * education
        + 0.14 * max(0.0, 1.0 - services_exposure),
    )
    behavioral = _score_behavior(candidate, intent, today)

    semantic_fit = min(1.0, 0.62 * semantic + 0.24 * title + 0.14 * years)
    return FeatureBreakdown(
        semantic_fit=semantic_fit,
        domain_fit=domain,
        production_experience=production,
        behavioral_score=behavioral,
        career_quality=career_quality,
        education_score=education,
        title_score=title,
        years_score=years,
        domain_hits=domain_hits,
        vector_hits=vector_hits,
        production_hits=production_hits,
        evaluation_hits=evaluation_hits,
        product_exposure=product_exposure,
        services_exposure=services_exposure,
    )
