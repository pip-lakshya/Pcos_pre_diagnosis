from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session
from typing import Literal
import re

from app.agent.research_provider import KnowledgeBaseProvider, ResearchProvider
from app.auth.security import get_current_user
from app.db.database import get_db
from app.db.models import Screening, User

router = APIRouter(prefix="/chat/research", tags=["research"])
_provider: ResearchProvider = KnowledgeBaseProvider()


def get_research_provider() -> ResearchProvider:
    return _provider


class ResearchTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=3000)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation: list[ResearchTurn] = Field(default_factory=list, max_length=12)


class ResearchResponse(BaseModel):
    answer: str


_FEATURE_LABELS = {
    "age": "age",
    "weight_kg": "weight input",
    "height_cm": "height input",
    "cycle_length_days": "cycle length",
    "cycle_irregular": "irregular cycles",
    "weight_gain": "reported weight gain",
    "hair_growth": "reported increased hair growth",
    "skin_darkening": "reported skin darkening",
    "hair_loss": "reported hair loss",
    "acne": "reported acne",
    "fast_food": "reported fast-food frequency",
    "regular_exercise": "reported exercise frequency",
    "hip_inch": "hip measurement input",
    "waist_inch": "waist measurement input",
    "bmi": "BMI input derived from height and weight",
    "waist_hip_ratio": "waist-to-hip measurement ratio",
}
_BOOLEAN_INPUTS = {
    "cycle_irregular", "weight_gain", "hair_growth", "skin_darkening",
    "hair_loss", "acne", "fast_food", "regular_exercise",
}
_DETAILS_QUERY = re.compile(
    r"\b(my (?:saved )?(?:details|answers|responses)|tell me my (?:details|answers)|show me my (?:details|answers)|"
    r"what did i (?:tell|answer|provide|say)|what did you (?:record|save)|details you (?:filled|recorded|saved)|"
    r"details i (?:gave|provided|filled)|what have i (?:filled|answered))\b|"
    r"(meri details|mere (?:screening )?details|mere jawab|maine kya (?:bhara|bataya|diya)|maine kya.{0,20}details|maine jo (?:bataya|diya))", re.IGNORECASE
)


def _saved_details_answer(question: str, screening: Screening) -> str | None:
    if not _DETAILS_QUERY.search(question):
        return None
    stored = screening.feature_state_json or {}
    features = stored.get("features", {})
    labels = {
        "age": "Age", "weight_kg": "Reported weight", "height_cm": "Reported height",
        "cycle_irregular": "Irregular cycles", "cycle_length_days": "Cycle length",
        "weight_gain": "Reported weight gain", "hair_growth": "Reported increased hair growth",
        "skin_darkening": "Reported skin darkening", "hair_loss": "Reported hair loss",
        "acne": "Reported acne", "fast_food": "Reported fast-food frequency",
        "regular_exercise": "Reported regular exercise", "hip_inch": "Reported hip measurement",
        "waist_inch": "Reported waist measurement", "bmi": "Calculated BMI",
        "waist_hip_ratio": "Calculated waist-to-hip ratio",
    }
    boolean_fields = _BOOLEAN_INPUTS
    values = []
    for key, label in labels.items():
        if key not in features:
            continue
        value = features[key]
        if key in {"hip_inch", "waist_inch", "waist_hip_ratio"} and value in (0, 0.0, None):
            continue
        if key in boolean_fields:
            value = "yes" if value in (1, True) else "no"
        elif key in {"weight_kg", "height_cm", "bmi"}:
            unit = {"weight_kg": " kg", "height_cm": " cm", "bmi": ""}[key]
            value = f"{value:g}{unit}"
        elif key == "hip_inch" or key == "waist_inch":
            value = f"{value:g} in"
        elif key == "waist_hip_ratio":
            value = f"{value:g}"
        values.append(f"{label}: {value}")
    risk = f"{screening.risk_probability * 100:.0f}% ({screening.risk_label} range)"
    details = "; ".join(values) if values else "No saved input details were available."
    if re.search(r"[\u0900-\u097f]|\b(meri|mere|maine|kya|jawab)\b", question.lower()):
        return (
            f"Aapke latest screening mein yeh details save hui thi: {details}. Model ka saved estimate {risk} tha. "
            "Yeh screening estimate hai, diagnosis nahi; apni health concerns ke liye doctor se salah lein. "
            "You can correct any detail in a new screening conversation."
        )
    return (
        f"Your latest saved screening recorded: {details}. The model's saved estimate was {risk}. "
        "This is a screening estimate, not a diagnosis; please consult a doctor about personal concerns. "
        "You can correct any detail in a new screening conversation."
    )


def build_screening_context(screening: Screening) -> dict:
    """Expose only the saved label and human-readable input names, never raw values."""
    stored = screening.feature_state_json or {}
    features = stored.get("features", {})
    importances = stored.get("feature_importances", {})
    top_names = [name for name, _ in sorted(importances.items(), key=lambda item: item[1], reverse=True)[:5]]
    return {
        "screening_level": screening.risk_label,
        "top_contributing_features": [_FEATURE_LABELS.get(name, name.replace("_", " ")) for name in top_names],
        "reported_positive_features_among_top_inputs": [
            _FEATURE_LABELS.get(name, name.replace("_", " "))
            for name in top_names
            if name in _BOOLEAN_INPUTS and features.get(name) in (1, True)
        ],
        "interpretation_limit": "Input importance is not causation; do not diagnose or change the saved result.",
    }


@router.post("", response_model=ResearchResponse)
def research(
    request: ResearchRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    provider: ResearchProvider = Depends(get_research_provider),
):
    screening = (
        db.query(Screening)
        .filter(Screening.user_id == user.id, Screening.completed.is_(True))
        .order_by(desc(Screening.created_at), desc(Screening.id))
        .first()
    )
    if screening is None:
        raise HTTPException(status_code=409, detail="Complete a screening before asking a personalized follow-up question.")
    context = build_screening_context(screening)
    saved_details = _saved_details_answer(request.question, screening)
    if saved_details:
        from app.agent.research_provider import PERSONALIZED_INSIGHTS_DISCLAIMER
        return ResearchResponse(answer=f"{saved_details}\n\n{PERSONALIZED_INSIGHTS_DISCLAIMER}")
    context["recent_conversation"] = [turn.model_dump() for turn in request.conversation]
    try:
        answer = provider.answer(request.question, context)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Research service failed: {error}")
    return ResearchResponse(answer=answer)
