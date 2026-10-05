from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.auth.security import get_current_user
from app.config import settings
from app.db.models import User
from app.services.magpie_tts import open_stream as open_magpie_stream

router = APIRouter(prefix="/tts", tags=["voice"])


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1800)


@router.get("/config")
def tts_config(_user: User = Depends(get_current_user)):
    voice = settings.magpie_tts_voice if settings.tts_provider == "magpie" else settings.edge_tts_voice
    return {"provider": settings.tts_provider, "voice": voice}


@router.post("")
async def edge_tts(request: TTSRequest, _user: User = Depends(get_current_user)):
    if settings.tts_provider == "magpie":
        if not settings.nvidia_api_key:
            raise HTTPException(status_code=503, detail="NVIDIA TTS is not configured")
        try:
            client, upstream = await open_magpie_stream(request.text)
        except Exception as error:
            raise HTTPException(status_code=502, detail="Magpie speech synthesis is unavailable right now") from error

        async def pcm_chunks():
            try:
                async for chunk in upstream.content.iter_chunked(8192):
                    yield chunk
            finally:
                upstream.release()
                await client.close()

        return StreamingResponse(
            pcm_chunks(),
            media_type="application/octet-stream",
            headers={"X-Audio-Format": "s16le; rate=22050; channels=1"},
        )

    if settings.tts_provider != "edge":
        raise HTTPException(status_code=409, detail="The browser speech provider does not use this endpoint")
    try:
        import edge_tts
    except ImportError:
        raise HTTPException(status_code=503, detail="Edge TTS is not installed")
    try:
        audio = BytesIO()
        stream = edge_tts.Communicate(request.text, settings.edge_tts_voice)
        async for item in stream.stream():
            if item.get("type") == "audio":
                audio.write(item["data"])
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Speech synthesis failed ({type(error).__name__})")
    if not audio.tell():
        raise HTTPException(status_code=502, detail="Speech synthesis returned no audio")
    return Response(content=audio.getvalue(), media_type="audio/mpeg")
