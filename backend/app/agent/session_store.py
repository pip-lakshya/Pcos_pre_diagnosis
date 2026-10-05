"""Per-session raw answers and Python-derived model features."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from uuid import uuid4

from app.config import settings

with Path(settings.features_path).open(encoding="utf-8") as feature_file:
    FEATURES = json.load(feature_file)

DERIVED_FEATURES = {"bmi", "waist_hip_ratio"}
COMPOSITE_FEATURES = {
    "androgen_score",
    "metabolic_score",
    "physical_risk",
    "cycle_severity",
}
RAW_FEATURES = [name for name in FEATURES if name not in DERIVED_FEATURES]
INTAKE_ORDER = [
    "age", "weight_kg", "height_cm", "cycle_irregular", "cycle_length_days",
    "weight_gain", "hair_growth", "skin_darkening", "hair_loss", "acne",
    "fast_food", "regular_exercise", "hip_inch", "waist_inch",
]
BOOLEAN_FEATURES = {
    "cycle_irregular",
    "weight_gain",
    "hair_growth",
    "skin_darkening",
    "hair_loss",
    "acne",
    "fast_food",
    "regular_exercise",
}
INTEGER_FEATURES = {"age", "cycle_length_days"}
OPTIONAL = {"hip_inch", "waist_inch", "waist_hip_ratio"}

if len(RAW_FEATURES) != 14 or not DERIVED_FEATURES.issubset(FEATURES):
    raise RuntimeError("Expected 14 raw fields and two Python-derived model features")


@dataclass
class Session:
    user_id: int
    values: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    screening_id: int | None = None
    skipped: set = field(default_factory=set)
    intake_turns: int = 0
    awaiting_confirmation: bool = False
    screening_started: bool = False


SESSIONS: dict[tuple[int, str], Session] = {}


def get_session(session_id: str | None, user_id: int):
    key = session_id or str(uuid4())
    session_key = (user_id, key)
    if session_key not in SESSIONS:
        SESSIONS[session_key] = Session(user_id=user_id)
    return key, SESSIONS[session_key]


def missing(session: Session) -> list[str]:
    """Return unanswered raw slots, including measurements users may skip."""
    return [
        name for name in INTAKE_ORDER
        if name not in session.values and name not in session.skipped
    ]


def merge_features(session: Session, updates: dict) -> set[str]:
    """Validate and merge non-null raw LLM answers, returning fields that changed."""
    unknown = set(updates) - set(RAW_FEATURES)
    if unknown:
        raise ValueError(f"Extractor returned non-raw or unknown features: {sorted(unknown)}")

    changed: set[str] = set()
    for name, value in updates.items():
        if value is None:
            continue
        if name in BOOLEAN_FEATURES:
            if type(value) is not bool:
                raise ValueError(f"Feature {name} must be a boolean")
            normalized = int(value)
        elif name in INTEGER_FEATURES:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Feature {name} must be an integer")
            if value < 0:
                raise ValueError(f"Feature {name} cannot be negative")
            normalized = value
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Feature {name} must be a finite number")
            if value <= 0:
                raise ValueError(f"Feature {name} must be greater than zero")
            normalized = float(value)

        if session.values.get(name) != normalized:
            changed.add(name)
        session.values[name] = normalized
        session.skipped.discard(name)

    return changed


def derive_features(session: Session) -> None:
    """Calculate backend-only BMI, ratio, and optional comparison annotations."""

    weight = session.values.get("weight_kg")
    height = session.values.get("height_cm")
    if weight is not None and height is not None:
        session.values["bmi"] = round(weight / (height / 100) ** 2, 2)

    hip = session.values.get("hip_inch")
    waist = session.values.get("waist_inch")
    if hip is not None and waist is not None:
        session.values["waist_hip_ratio"] = round(waist / hip, 3)

    # Research/evaluation composites are Python-only session annotations. They are
    # intentionally absent from the canonical model feature JSON and never enter
    # the extraction schema or the 16-column inference row.
    components = ("hair_growth", "skin_darkening", "acne", "hair_loss")
    if all(name in session.values for name in components):
        session.values["androgen_score"] = sum(session.values[name] for name in components)

    if all(name in session.values for name in ("weight_gain", "fast_food", "bmi")):
        session.values["metabolic_score"] = (
            session.values["weight_gain"]
            + session.values["fast_food"]
            + int(session.values["bmi"] > 25)
        )

    if all(name in session.values for name in ("bmi", "waist_hip_ratio")):
        session.values["physical_risk"] = round(
            session.values["bmi"] * session.values["waist_hip_ratio"], 3
        )

    if all(name in session.values for name in ("cycle_irregular", "cycle_length_days")):
        session.values["cycle_severity"] = (
            session.values["cycle_irregular"] * session.values["cycle_length_days"]
        )
