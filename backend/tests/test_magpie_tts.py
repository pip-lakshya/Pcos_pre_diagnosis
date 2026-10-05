import unittest
from unittest.mock import patch
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.api.tts import TTSRequest, edge_tts
from app.config import settings
from app.services import magpie_tts


class FakeResponse:
    status = 200
    def __await__(self):
        async def result():
            return self
        return result().__await__()

    def release(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def read(self):
        return b"RIFFmock-wav"


class FakeClientSession:
    latest = None

    def __init__(self, **kwargs):
        self.options = kwargs
        self.closed = False
        FakeClientSession.latest = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def post(self, url, data, headers):
        self.url = url
        self.form = data
        self.headers = headers
        return FakeResponse()

    async def close(self):
        self.closed = True


class FakePcmContent:
    async def iter_chunked(self, size):
        yield b"pcm-first"
        yield b"pcm-second"


class FakeUpstreamResponse:
    content = FakePcmContent()

    def release(self):
        return None


class MagpieTtsTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthesis_sends_multipart_to_nvidia_and_returns_wav(self):
        with patch.object(settings, "nvidia_api_key", "test-key"), patch.object(
            settings, "magpie_tts_url", "https://magpie.test/v1/audio/synthesize"
        ), patch.object(settings, "magpie_tts_voice", "Magpie-Multilingual.EN-US.Aria"), patch.object(
            magpie_tts.aiohttp, "ClientSession", FakeClientSession
        ):
            audio = await magpie_tts.synthesize("Hello in English.")

        request = FakeClientSession.latest
        fields = {field[0]["name"]: field[2] for field in request.form._fields}
        self.assertEqual(audio, b"RIFFmock-wav")
        self.assertEqual(request.url, "https://magpie.test/v1/audio/synthesize")
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        self.assertEqual(fields["voice"], "Magpie-Multilingual.EN-US.Aria")
        self.assertEqual(fields["language"], "en-US")
        self.assertEqual(fields["text"], "Hello in English.")
        self.assertEqual(fields["encoding"], "LINEAR_PCM")

    async def test_streaming_mode_uses_online_pcm_endpoint(self):
        with patch.object(settings, "nvidia_api_key", "test-key"), patch.object(
            settings, "magpie_tts_url", "https://magpie.test/v1/audio/synthesize"
        ), patch.object(magpie_tts.aiohttp, "ClientSession", FakeClientSession):
            client, response = await magpie_tts.open_stream("Quick response")
        self.assertEqual(client.url, "https://magpie.test/v1/audio/synthesize_online")
        self.assertEqual(client.headers["Authorization"], "Bearer test-key")
        self.assertEqual(client.form._fields[0][2], "Quick response")
        response.release()
        await client.close()

    async def test_authenticated_tts_route_streams_pcm_response(self):
        client = FakeClientSession()
        upstream = FakeUpstreamResponse()
        with patch.object(settings, "tts_provider", "magpie"), patch.object(
            settings, "nvidia_api_key", "test-key"
        ), patch("app.api.tts.open_magpie_stream", new=AsyncMock(return_value=(client, upstream))):
            response = await edge_tts(TTSRequest(text="Hello"), object())
            chunks = [chunk async for chunk in response.body_iterator]
        self.assertEqual(response.media_type, "application/octet-stream")
        self.assertEqual(response.headers["x-audio-format"], "s16le; rate=22050; channels=1")
        self.assertEqual(chunks, [b"pcm-first", b"pcm-second"])


if __name__ == "__main__":
    unittest.main()
