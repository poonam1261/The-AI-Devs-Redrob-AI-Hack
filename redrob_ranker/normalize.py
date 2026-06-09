from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedCandidate:
    candidate_id: str
    title: str
    headline: str
    summary: str
    location: str
    country: str
    years: float
    current_company: str
    current_industry: str
    current_company_size: str
    career: tuple[dict[str, Any], ...]
    education: tuple[dict[str, Any], ...]
    skills: tuple[dict[str, Any], ...]
    signals: dict[str, Any]
    skill_names: tuple[str, ...]
    profile_text: str
    career_text: str
    skill_text: str
    full_text: str


def _join(values: list[str]) -> str:
    return " ".join(v.strip() for v in values if v and v.strip())


def normalize_candidate(raw: dict[str, Any]) -> NormalizedCandidate:
    profile = raw.get("profile", {})
    career = tuple(raw.get("career_history", []) or [])
    education = tuple(raw.get("education", []) or [])
    skills = tuple(raw.get("skills", []) or [])
    signals = dict(raw.get("redrob_signals", {}) or {})
    skill_names = tuple(str(skill.get("name", "")).strip() for skill in skills if skill.get("name"))

    profile_text = _join(
        [
            profile.get("headline", ""),
            profile.get("summary", ""),
            profile.get("current_title", ""),
            profile.get("current_industry", ""),
            profile.get("location", ""),
        ]
    )
    career_text = _join(
        [
            _join(
                [
                    item.get("title", ""),
                    item.get("company", ""),
                    item.get("industry", ""),
                    item.get("description", ""),
                ]
            )
            for item in career
        ]
    )
    skill_text = _join(
        [
            f"{skill.get('name', '')} {skill.get('proficiency', '')} {skill.get('duration_months', '')}"
            for skill in skills
        ]
    )
    edu_text = _join(
        [
            _join([item.get("degree", ""), item.get("field_of_study", ""), item.get("institution", "")])
            for item in education
        ]
    )
    full_text = _join([profile_text, career_text, skill_text, edu_text])

    return NormalizedCandidate(
        candidate_id=raw["candidate_id"],
        title=str(profile.get("current_title", "")),
        headline=str(profile.get("headline", "")),
        summary=str(profile.get("summary", "")),
        location=str(profile.get("location", "")),
        country=str(profile.get("country", "")),
        years=float(profile.get("years_of_experience", 0.0) or 0.0),
        current_company=str(profile.get("current_company", "")),
        current_industry=str(profile.get("current_industry", "")),
        current_company_size=str(profile.get("current_company_size", "")),
        career=career,
        education=education,
        skills=skills,
        signals=signals,
        skill_names=skill_names,
        profile_text=profile_text,
        career_text=career_text,
        skill_text=skill_text,
        full_text=full_text,
    )
