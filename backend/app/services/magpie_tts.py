"""NVIDIA hosted Magpie TTS adapter."""

import aiohttp

from app.config import settings


async def synthesize(text: str) -> bytes:
    """Return a WAV byte stream from the configured NVIDIA Magpie endpoint."""
    fields = {
        "text": text,
        "language": "en-US",
        "voice": settings.magpie_tts_voice,
        "encoding": "LINEAR_PCM",
        "sample_rate_hz": "22050",
    }
    form = aiohttp.FormData(default_to_multipart=True)
    for name, value in fields.items():
        form.add_field(name, value)
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as client:
        async with client.post(
            settings.magpie_tts_url,
            data=form,
            headers={
                "Authorization": f"Bearer {settings.nvidia_api_key}",
                "Accept": "audio/wav",
            },
        ) as response:
            if response.status >= 400:
                raise RuntimeError(f"Magpie TTS returned HTTP {response.status}")
            audio = await response.read()
    if not audio:
        raise RuntimeError("Magpie returned an empty audio response")
    return audio


async def open_stream(text: str) -> tuple[aiohttp.ClientSession, aiohttp.ClientResponse]:
    """Open NVIDIA's chunked raw-PCM endpoint for low-delay playback."""
    fields = {
        "text": text,
        "language": "en-US",
        "voice": settings.magpie_tts_voice,
        "encoding": "LINEAR_PCM",
        "sample_rate_hz": "22050",
    }
    form = aiohttp.FormData(default_to_multipart=True)
    for name, value in fields.items():
        form.add_field(name, value)
    stream_url = f"{settings.magpie_tts_url.rsplit('/', 1)[0]}/synthesize_online"
    client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90, connect=10))
    try:
        response = await client.post(
            stream_url,
            data=form,
            headers={"Authorization": f"Bearer {settings.nvidia_api_key}"},
        )
        if response.status >= 400:
            status = response.status
            response.release()
            await client.close()
            raise RuntimeError(f"Magpie TTS returned HTTP {status}")
        return client, response
    except Exception:
        if not client.closed:
            await client.close()
        raise
