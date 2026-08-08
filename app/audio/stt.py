"""Incremental faster-whisper transcription adapter."""

from collections.abc import Iterable

import numpy as np
import webrtcvad
from faster_whisper import WhisperModel

from app.config import settings


class StreamingTranscriber:
    """Accumulates PCM frames received on one persistent WebSocket session."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._model: WhisperModel | None = None
        self._vad = webrtcvad.Vad(2)
        self._vad_remainder = b""

    def add_audio(self, chunk: bytes) -> None:
        """Append 16 kHz mono signed-16-bit PCM data without persisting raw audio."""
        self._chunks.append(chunk)

    def contains_speech(self, chunk: bytes, sample_rate: int = 16_000) -> bool:
        """Return whether a chunk contains speech in a valid WebRTC VAD frame.

        Browser capture callbacks are not guaranteed to be 10/20/30 ms long,
        so retain the partial tail and examine contiguous 30 ms PCM frames.
        """
        frame_bytes = sample_rate * 30 // 1000 * 2
        audio = self._vad_remainder + chunk
        speech = False
        offset = 0
        while offset + frame_bytes <= len(audio):
            speech = self._vad.is_speech(audio[offset : offset + frame_bytes], sample_rate) or speech
            offset += frame_bytes
        self._vad_remainder = audio[offset:]
        return speech

    def transcribe_final(self) -> str:
        """Transcribe the accumulated current utterance and discard its audio data."""
        audio = b"".join(self._chunks)
        self._chunks.clear()
        self._vad_remainder = b""
        if not audio:
            return ""
        if self._model is None:
            self._model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments: Iterable[object]
        segments, _ = self._model.transcribe(
            samples,
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(getattr(segment, "text", "").strip() for segment in segments).strip()
