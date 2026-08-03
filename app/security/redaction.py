"""Minimal deterministic PII redaction for persisted conversation logs."""

import re

_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_ADDRESS = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,4}\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln)\b",
    re.I,
)


def redact_pii(text: str) -> str:
    """Replace common payment, SSN, and street-address patterns before logging."""
    text = _CARD.sub("[REDACTED_CARD]", text)
    text = _SSN.sub("[REDACTED_SSN]", text)
    return _ADDRESS.sub("[REDACTED_ADDRESS]", text)
