from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.security import get_current_user
from app.db.database import get_db
from app.db.models import Screening, User

router = APIRouter(prefix="/screenings", tags=["screenings"])


class ScreeningHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    risk_probability: float
    risk_label: str
    created_at: datetime
    completed: bool


@router.get("/me", response_model=list[ScreeningHistoryItem])
def my_screenings(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = db.scalars(
        select(Screening)
        .where(Screening.user_id == user.id, Screening.completed.is_(True))
        .order_by(Screening.created_at.desc(), Screening.id.desc())
    ).all()
    return rows
