"""Incremental faster-whisper transcription adapter."""

from collections.abc import Iterable

import numpy as np
from faster_whisper import WhisperModel

from app.config import settings


class StreamingTranscriber:
    """Accumulates PCM frames received on one persistent WebSocket session."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._model: WhisperModel | None = None

    def add_audio(self, chunk: bytes) -> None:
        """Append 16 kHz mono signed-16-bit PCM data without persisting raw audio."""
        self._chunks.append(chunk)

    def transcribe_final(self) -> str:
        """Transcribe the accumulated current utterance and discard its audio data."""
        audio = b"".join(self._chunks)
        self._chunks.clear()
        if not audio:
            return ""
        if self._model is None:
            self._model = WhisperModel(settings.whisper_model, device="auto", compute_type="int8")
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments: Iterable[object]
        segments, _ = self._model.transcribe(samples, language="en", vad_filter=True)
        return " ".join(getattr(segment, "text", "").strip() for segment in segments).strip()
