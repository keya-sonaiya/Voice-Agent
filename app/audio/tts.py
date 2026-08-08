"""Interruptible speech synthesis exposed as WebSocket audio chunks."""

import asyncio
from collections.abc import AsyncIterator
from io import BytesIO

import soundfile as sf
from TTS.api import TTS

from app.config import settings


class InterruptibleSynthesizer:
    """Produces WAV data in chunks and can be cancelled immediately for barge-in."""

    def __init__(self) -> None:
        self._cancelled = asyncio.Event()
        self._engine: TTS | None = None

    def interrupt(self) -> None:
        """Signal the active output stream to stop before its next chunk."""
        self._cancelled.set()

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield synthesized WAV chunks; caller interruption prevents subsequent chunks."""
        self._cancelled.clear()
        if settings.tts_provider != "coqui":
            raise RuntimeError("Only the configured local Coqui provider is available in this demo.")
        if self._engine is None:
            self._engine = await asyncio.to_thread(
                TTS, model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False
            )
        wav = await asyncio.to_thread(self._engine.tts, text=text)
        buffer = BytesIO()
        sf.write(buffer, wav, 22050, format="WAV")
        data = buffer.getvalue()
        for index in range(0, len(data), 4096):
            if self._cancelled.is_set():
                break
            yield data[index : index + 4096]
            await asyncio.sleep(0)
