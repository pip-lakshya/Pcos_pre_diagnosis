"""Structured multiple-choice screening using the existing risk model."""
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent.session_store import BOOLEAN_FEATURES, INTAKE_ORDER, OPTIONAL, derive_features
from app.auth.security import get_current_user
from app.db.database import get_db
from app.db.models import Screening, User
from app.ml.predictor import load_assets, predict

router = APIRouter(prefix="/screenings", tags=["screenings"])


class ManualScreeningRequest(BaseModel):
    features: dict[str, Any] = Field(description="Self-reported raw screening answers")
    unknown_fields: list[str] = Field(default_factory=list)


@router.post("/manual")
def submit_manual_screening(
    request: ManualScreeningRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    unknown = set(request.features) - set(INTAKE_ORDER)
    if unknown:
        raise HTTPException(status_code=422, detail="Unknown screening fields were provided")
    invalid_unknown = set(request.unknown_fields) - BOOLEAN_FEATURES
    if invalid_unknown:
        raise HTTPException(status_code=422, detail="Unknown answers are supported only for yes/no questions")

    missing = set(INTAKE_ORDER) - set(request.features) - OPTIONAL - set(request.unknown_fields)
    if missing:
        raise HTTPException(status_code=422, detail="Please answer all required screening questions")

    values: dict[str, int | float] = {}
    for name, value in request.features.items():
        if name in BOOLEAN_FEATURES:
            if type(value) is not bool:
                raise HTTPException(status_code=422, detail=f"Invalid answer for {name}")
            values[name] = int(value)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(status_code=422, detail=f"Invalid numeric answer for {name}")
            if value <= 0:
                if name in OPTIONAL and value == 0:
                    values[name] = 0.0
                    continue
                raise HTTPException(status_code=422, detail=f"Invalid numeric answer for {name}")
            values[name] = float(value)

    # The trained model has no distinct representation for an unsure answer.
    # Keep that uncertainty in the saved record and use the binary baseline (0)
    # for inference, matching the model's accepted yes/no input shape.
    for name in request.unknown_fields:
        values[name] = 0

    class SessionValues:
        pass

    session = SessionValues()
    session.values = values
    derive_features(session)
    try:
        _, model_features = load_assets()
        payload = {name: values.get(name, 0.0) for name in model_features}
        output = predict(payload)
    except Exception as error:
        raise HTTPException(status_code=500, detail="Screening estimate could not be calculated") from error

    saved_features = {**values, **{name: values.get(name, 0.0) for name in OPTIONAL}}
    screening = Screening(
        user_id=user.id,
        feature_state_json={
            "features": saved_features,
            "unknown_fields": request.unknown_fields,
            "feature_importances": output.feature_importances,
            "agreement_score": output.agreement_score,
            "model_probabilities": output.model_probabilities,
            "screening_mode": "multiple_choice",
        },
        risk_probability=output.probability,
        risk_label=output.risk_label,
        completed=True,
    )
    db.add(screening)
    db.commit()
    db.refresh(screening)
    return {
        "screening_id": screening.id,
        "features": saved_features,
        "unknown_fields": request.unknown_fields,
        "prediction": output,
    }
