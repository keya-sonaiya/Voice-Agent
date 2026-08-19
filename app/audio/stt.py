"""Incremental faster-whisper transcription adapter."""

import os
import site
from collections.abc import Iterable
from pathlib import Path
from time import monotonic

import numpy as np
import webrtcvad  # type: ignore[import-untyped]
from faster_whisper import WhisperModel

from app.call_logging import call_exception, call_log, duration_ms
from app.config import settings

_DLL_DIRECTORY_HANDLES: list[object] = []


def _configure_windows_cuda_dlls() -> None:
    """Make pip-installed CUDA 12/cuDNN DLLs discoverable by CTranslate2.

    NVIDIA's runtime wheels place binaries below ``site-packages/nvidia``;
    Windows does not search those folders unless they are added explicitly.
    Keep handles alive for the process lifetime as required by
    ``os.add_dll_directory``.
    """
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    dll_directories: list[str] = []
    for site_package in site.getsitepackages():
        root = Path(site_package) / "nvidia"
        for directory in (root / "cuda_nvrtc" / "bin", root / "cublas" / "bin", root / "cudnn" / "bin"):
            if directory.is_dir():
                directory_text = str(directory)
                dll_directories.append(directory_text)
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory_text))
    if dll_directories:
        # CTranslate2 resolves CUDA libraries by name through PATH at first
        # inference, whereas ctypes honors add_dll_directory directly.
        os.environ["PATH"] = os.pathsep.join([*dll_directories, os.environ.get("PATH", "")])


class StreamingTranscriber:
    """Accumulates PCM frames received on one persistent WebSocket session."""

    def __init__(self, session_id: str = "unknown") -> None:
        self.session_id = session_id
        self._chunks: list[bytes] = []
        self._model: WhisperModel | None = None
        self._vad = webrtcvad.Vad(2)
        self._vad_remainder = b""
        self.audio_frame_count = 0
        self.audio_byte_count = 0

    def add_audio(self, chunk: bytes) -> None:
        """Append 16 kHz mono signed-16-bit PCM data without persisting raw audio."""
        self._chunks.append(chunk)
        self.audio_frame_count += 1
        self.audio_byte_count += len(chunk)

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
        if speech:
            call_log(
                self.session_id,
                "AUDIO",
                "speech_detected",
                details={"frame_count": self.audio_frame_count, "accumulated_bytes": self.audio_byte_count},
            )
        return speech

    def transcribe_final(self) -> str:
        """Transcribe the accumulated current utterance and discard its audio data."""
        started = monotonic()
        audio = b"".join(self._chunks)
        self._chunks.clear()
        self._vad_remainder = b""
        if not audio:
            call_log(self.session_id, "STT", "empty_audio")
            return ""
        call_log(
            self.session_id,
            "STT",
            "start",
            details={"audio_bytes": len(audio), "frame_count": self.audio_frame_count},
        )
        try:
            if self._model is None:
                call_log(
                    self.session_id,
                    "STT",
                    "model_initializing",
                    details={"model": settings.whisper_model, "device": settings.whisper_device},
                )
                _configure_windows_cuda_dlls()
                self._model = WhisperModel(
                    settings.whisper_model,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                )
                call_log(self.session_id, "STT", "model_initialized")
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            segments: Iterable[object]
            segments, _ = self._model.transcribe(
                samples,
                language="en",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            transcript = " ".join(getattr(segment, "text", "").strip() for segment in segments).strip()
        except Exception:
            call_exception(self.session_id, "STT", "failed")
            raise
        finally:
            self.audio_frame_count = 0
            self.audio_byte_count = 0
        if transcript:
            call_log(
                self.session_id,
                "STT",
                "complete",
                duration=duration_ms(started),
                details={"transcript": transcript, "transcript_length": len(transcript)},
            )
        else:
            call_log(self.session_id, "STT", "empty_transcript", duration=duration_ms(started))
        return transcript
