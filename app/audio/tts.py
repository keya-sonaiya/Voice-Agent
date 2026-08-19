"""Interruptible Coqui and ElevenLabs synthesis for WebSocket playback."""

import asyncio
from collections.abc import AsyncIterator
from io import BytesIO

import soundfile as sf
from elevenlabs.client import ElevenLabs
from TTS.api import TTS

from app.call_logging import call_exception, call_log
from app.config import settings

CHUNK_SIZE = 4096


class InterruptibleSynthesizer:
    """Produce complete browser-decodable audio containers in interruptible WS chunks."""

    def __init__(self) -> None:
        self._cancelled = asyncio.Event()
        self._coqui_engine: TTS | None = None
        self._elevenlabs_client: ElevenLabs | None = None

    @property
    def output_format(self) -> str:
        """Return the container MIME type the browser should decode after ``tts_complete``."""
        return "audio/mpeg" if settings.tts_provider == "elevenlabs" else "audio/wav"

    def interrupt(self) -> None:
        """Stop streaming additional chunks as soon as a caller barges in or disconnects."""
        self._cancelled.set()

    def _synthesize_coqui(self, text: str) -> bytes:
        if self._coqui_engine is None:
            self._coqui_engine = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        wav = self._coqui_engine.tts(text=text)
        sample_rate = int(getattr(self._coqui_engine.synthesizer, "output_sample_rate", 22050))
        buffer = BytesIO()
        sf.write(buffer, wav, sample_rate, format="WAV")
        return buffer.getvalue()

    def _synthesize_elevenlabs(self, text: str) -> bytes:
        if self._elevenlabs_client is None:
            # Settings validation guarantees a key before application startup reaches this point.
            self._elevenlabs_client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        chunks = self._elevenlabs_client.text_to_speech.convert(
            voice_id=settings.elevenlabs_voice_id,
            text=text,
            model_id=settings.elevenlabs_model_id,
            # A complete MP3 container is browser-decodable through AudioContext.decodeAudioData.
            output_format="mp3_44100_128",
        )
        return b"".join(chunks)

    async def stream(self, text: str, session_id: str = "unknown") -> AsyncIterator[bytes]:
        """Synthesize one complete WAV/MP3 asset and send it in ordered WS binary chunks."""
        self._cancelled.clear()
        call_log(
            session_id,
            "TTS",
            "provider_selected",
            details={"provider": settings.tts_provider, "format": self.output_format, "text_length": len(text)},
        )
        started = asyncio.get_running_loop().time()
        call_log(session_id, "TTS", "synthesis_start", details={"provider": settings.tts_provider})
        try:
            if settings.tts_provider == "elevenlabs":
                data = await asyncio.to_thread(self._synthesize_elevenlabs, text)
            else:
                data = await asyncio.to_thread(self._synthesize_coqui, text)
            if not data:
                raise RuntimeError("TTS provider returned no audio bytes.")
        except Exception:
            call_exception(session_id, "TTS", "synthesis_failed", details={"provider": settings.tts_provider})
            raise
        call_log(
            session_id,
            "TTS",
            "synthesis_complete",
            duration=int((asyncio.get_running_loop().time() - started) * 1000),
            details={"provider": settings.tts_provider, "output_bytes": len(data)},
        )
        for index in range(0, len(data), CHUNK_SIZE):
            if self._cancelled.is_set():
                call_log(session_id, "TTS", "interrupted", details={"sent_bytes": index})
                return
            yield data[index : index + CHUNK_SIZE]
            await asyncio.sleep(0)
