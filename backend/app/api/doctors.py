from fastapi import APIRouter, Depends, HTTPException

from app.auth.security import get_current_user
from app.db.models import User
from app.models.schemas import NearbyDoctorRequest, NearbyDoctorResponse
from app.services.osm_doctors import OpenStreetMapDoctorFinder, finder

router = APIRouter(prefix="/doctors", tags=["doctors"])


def get_doctor_finder() -> OpenStreetMapDoctorFinder:
    return finder


@router.post("/nearby", response_model=NearbyDoctorResponse)
async def nearby_doctors(
    request: NearbyDoctorRequest,
    _user: User = Depends(get_current_user),
    doctor_finder: OpenStreetMapDoctorFinder = Depends(get_doctor_finder),
):
    try:
        result = await doctor_finder.search(lat=request.lat, long=request.long, query=request.query)
        if result["lat"] is None or result["long"] is None:
            raise HTTPException(
                status_code=404,
                detail="I could not locate that city or postal code. Check it or try a nearby larger city.",
            )
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "The doctor search could not reach OpenStreetMap right now. This is usually temporary. "
                "Please retry; if it keeps failing, check that the backend host can access the internet over HTTPS and resolve DNS."
            ),
        ) from error
    return result
