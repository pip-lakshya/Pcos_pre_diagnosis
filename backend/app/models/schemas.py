from typing import Any
from pydantic import BaseModel, Field, model_validator

class PredictRequest(BaseModel):
    features: dict[str, Any]
class PredictResponse(BaseModel):
    probability: float
    risk_label: str
    feature_importances: dict[str, float]
    agreement_score: float
    model_probabilities: dict[str, float]
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)
class ChatResponse(BaseModel):
    session_id: str
    reply: str
    collected: dict[str, Any]
    missing: list[str]
    complete: bool
    prediction: PredictResponse | None = None
    screening_id: int | None = None
    awaiting_confirmation: bool = False


class NearbyDoctorRequest(BaseModel):
    lat: float | None = Field(default=None, ge=-90, le=90)
    long: float | None = Field(default=None, ge=-180, le=180)
    query: str | None = Field(default=None, min_length=2, max_length=200)

    @model_validator(mode="after")
    def validate_location(self):
        has_any_coordinate = self.lat is not None or self.long is not None
        has_both_coordinates = self.lat is not None and self.long is not None
        if bool(self.query) == has_any_coordinate:
            raise ValueError("Provide either query or both lat and long")
        if has_any_coordinate and not has_both_coordinates:
            raise ValueError("Both lat and long are required")
        if self.query is not None:
            self.query = self.query.strip()
            if len(self.query) < 2:
                raise ValueError("Enter a city or postal code")
        return self


class DoctorResult(BaseModel):
    name: str
    address: str
    phone: str
    lat: float
    long: float
    directions_url: str


class NearbyDoctorResponse(BaseModel):
    location: str | None
    lat: float | None
    long: float | None
    radius_km: int
    radius_widened: bool
    doctors: list[DoctorResult]
