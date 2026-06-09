from __future__ import annotations

from .jd import JobIntent
from .normalize import NormalizedCandidate
from .retrieval import count_phrase_hits
from .scoring import RankedCandidate


def _matched_terms(text: str, phrases: tuple[str, ...], limit: int = 4) -> list[str]:
    low = text.lower()
    hits = []
    for phrase in phrases:
        if phrase.lower() in low and phrase not in hits:
            hits.append(phrase)
        if len(hits) >= limit:
            break
    return hits


def _skill_names(candidate: NormalizedCandidate, limit: int = 4) -> list[str]:
    relevant = []
    preferred = {
        "python",
        "nlp",
        "llms",
        "rag",
        "semantic search",
        "recommendation systems",
        "learning to rank",
        "faiss",
        "elasticsearch",
        "pytorch",
        "scikit-learn",
        "fine-tuning llms",
    }
    for skill in candidate.skills:
        name = str(skill.get("name", ""))
        if name.lower() in preferred:
            relevant.append(name)
    return relevant[:limit]


def build_explanation(item: RankedCandidate, intent: JobIntent) -> dict[str, object]:
    candidate = item.candidate
    features = item.features
    signals = candidate.signals
    domain_terms = _matched_terms(candidate.full_text, intent.domain_terms, 4)
    vector_terms = _matched_terms(candidate.full_text, intent.vector_terms, 3)
    prod_terms = _matched_terms(candidate.career_text, intent.production_terms, 3)
    eval_terms = _matched_terms(candidate.career_text, intent.evaluation_terms, 2)
    skills = _skill_names(candidate)

    strengths: list[str] = [
        f"{candidate.title} with {candidate.years:.1f} years of experience",
    ]
    if domain_terms:
        strengths.append("career/profile evidence for " + ", ".join(domain_terms))
    if vector_terms:
        strengths.append("vector/search stack evidence: " + ", ".join(vector_terms))
    if prod_terms:
        strengths.append("production language in career history: " + ", ".join(prod_terms))
    if eval_terms:
        strengths.append("ranking evaluation evidence: " + ", ".join(eval_terms))
    if skills:
        strengths.append("listed relevant skills: " + ", ".join(skills))
    if features.product_exposure > 0:
        strengths.append(f"product-company exposure across {features.product_exposure:.0%} of listed roles")

    weaknesses: list[str] = []
    if not domain_terms:
        weaknesses.append("limited explicit search/retrieval evidence")
    if candidate.years < intent.preferred_experience_min or candidate.years > intent.preferred_experience_max:
        weaknesses.append("outside the JD's preferred 5-9 year experience band")
    if features.services_exposure >= 0.75:
        weaknesses.append("career is heavily services-oriented")
    if item.fraud.flags:
        weaknesses.extend(item.fraud.flags[:2])
    notice = int(signals.get("notice_period_days") or 180)
    if notice > 60:
        weaknesses.append(f"{notice}-day notice period raises availability risk")
    if float(signals.get("recruiter_response_rate") or 0.0) < 0.35:
        weaknesses.append("low recruiter response rate")

    behavioral = (
        f"Open to work: {bool(signals.get('open_to_work_flag'))}; "
        f"response rate {float(signals.get('recruiter_response_rate') or 0.0):.2f}; "
        f"interview completion {float(signals.get('interview_completion_rate') or 0.0):.2f}; "
        f"notice period {notice} days; last active {signals.get('last_active_date', 'unknown')}."
    )

    recruiter_summary = make_csv_reason(item, intent)
    return {
        "candidate_id": candidate.candidate_id,
        "final_score": round(item.final_score, 6),
        "ranking_position": item.rank,
        "score_breakdown": {
            "base_score": round(item.base_score, 6),
            "fraud_penalty": round(item.fraud.penalty, 6),
            **{key: round(value, 6) if isinstance(value, float) else value for key, value in features.as_dict().items()},
        },
        "strengths": strengths[:6],
        "weaknesses": weaknesses[:5],
        "behavioral_assessment": behavioral,
        "recruiter_summary": recruiter_summary,
    }


def make_csv_reason(item: RankedCandidate, intent: JobIntent) -> str:
    candidate = item.candidate
    signals = candidate.signals
    domain_terms = _matched_terms(candidate.full_text, intent.domain_terms, 3)
    vector_terms = _matched_terms(candidate.full_text, intent.vector_terms, 2)
    eval_count = count_phrase_hits(candidate.career_text, intent.evaluation_terms)
    notice = int(signals.get("notice_period_days") or 180)
    response = float(signals.get("recruiter_response_rate") or 0.0)

    evidence_parts = []
    if domain_terms:
        evidence_parts.append(", ".join(domain_terms))
    if vector_terms:
        evidence_parts.append(", ".join(vector_terms))
    if eval_count:
        evidence_parts.append("ranking evaluation evidence")
    evidence = "; ".join(evidence_parts) if evidence_parts else "adjacent ML/software evidence"

    concern = ""
    if item.fraud.penalty >= 0.18:
        concern = f" Concern: {item.fraud.flags[0]}."
    elif notice > 60:
        concern = f" Concern: {notice}-day notice period."
    elif response < 0.35:
        concern = f" Concern: recruiter response rate is {response:.2f}."

    return (
        f"{candidate.title} with {candidate.years:.1f} yrs in {candidate.current_industry}; "
        f"matches JD through {evidence}. Behavioral fit: response {response:.2f}, "
        f"last active {signals.get('last_active_date', 'unknown')}, notice {notice} days.{concern}"
    )
