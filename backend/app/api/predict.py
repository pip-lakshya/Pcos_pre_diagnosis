from fastapi import APIRouter, HTTPException
from app.models.schemas import PredictRequest, PredictResponse
from app.ml.predictor import predict
router=APIRouter()
@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try: return predict(request.features)
    except (ValueError, TypeError) as e: raise HTTPException(422,str(e))
