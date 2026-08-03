"""Transcript validation before any untrusted text reaches a model."""

import re

MAX_TRANSCRIPT_LENGTH = 2_000
_INSTRUCTION_PATTERNS = (
    r"ignore (all |any |the )?(previous|prior) instructions",
    r"you are now",
    r"system prompt",
    r"tool schemas?",
    r"reveal .*prompt",
)


class UnsafeTranscriptError(ValueError):
    """Raised when a transcript resembles a prompt-exfiltration attack."""


def validate_transcript(transcript: str) -> str:
    """Return normalized caller text or reject oversized/instruction-like content."""
    text = transcript.strip()
    if not text:
        raise UnsafeTranscriptError("Transcript must not be empty.")
    if len(text) > MAX_TRANSCRIPT_LENGTH:
        raise UnsafeTranscriptError("Transcript exceeds the per-turn size limit.")
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _INSTRUCTION_PATTERNS):
        raise UnsafeTranscriptError("Suspicious instruction-like transcript rejected.")
    return text
