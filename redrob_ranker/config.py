from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "weights": {
        "semantic_fit": 0.40,
        "domain_fit": 0.20,
        "production_experience": 0.15,
        "behavioral_score": 0.15,
        "career_quality": 0.10,
    },
    "fraud": {
        "max_penalty": 0.75,
        "expert_zero_duration_penalty": 0.08,
        "unsupported_ai_claim_penalty": 0.18,
        "timeline_penalty": 0.18,
        "nontechnical_ai_penalty": 0.25,
        "skill_inflation_penalty": 0.16,
        "services_only_penalty": 0.10,
    },
    "runtime": {
        "shortlist_size": 100,
        "explanation_json": "explanations.json",
    },
}


@dataclass(frozen=True)
class RankerConfig:
    weights: dict[str, float]
    fraud: dict[str, float]
    runtime: dict[str, Any]


def _coerce_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml(path: Path) -> dict[str, Any]:
    """Tiny YAML subset parser used only if PyYAML is unavailable."""
    parsed: dict[str, Any] = {}
    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section = line[:-1].strip()
            parsed[section] = {}
            continue
        if section and ":" in line:
            key, value = line.strip().split(":", 1)
            parsed[section][key.strip()] = _coerce_scalar(value)
    return parsed


def load_config(path: str | Path = "config/weights.yaml") -> RankerConfig:
    data = DEFAULT_CONFIG.copy()
    cfg_path = Path(path)
    if cfg_path.exists():
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            loaded = _simple_yaml(cfg_path)
        for section, values in loaded.items():
            if isinstance(values, dict):
                merged = dict(data.get(section, {}))
                merged.update(values)
                data[section] = merged
            else:
                data[section] = values
    return RankerConfig(
        weights={k: float(v) for k, v in data["weights"].items()},
        fraud={k: float(v) for k, v in data["fraud"].items()},
        runtime=dict(data["runtime"]),
    )
