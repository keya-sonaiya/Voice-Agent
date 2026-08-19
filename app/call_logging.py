"""Safe, correlated operational logging for live call processing."""

import logging
from collections.abc import Mapping
from time import monotonic
from typing import Any

from app.config import settings
from app.security.redaction import redact_pii

LOGGER = logging.getLogger("app.calls")


def duration_ms(started: float) -> int:
    """Return an integer elapsed time suitable for structured call logs."""
    return int((monotonic() - started) * 1000)


def _safe_value(value: Any) -> str:
    """Render small, secret-free log details without retaining raw caller content."""
    if isinstance(value, str):
        return redact_pii(value)[:160]
    return str(value)[:160]


def call_log(
    session_id: str,
    stage: str,
    event: str,
    *,
    level: int = logging.INFO,
    duration: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Emit the standard call trace format, retaining warnings/errors when debug is off."""
    if not settings.debug_logging and level < logging.WARNING:
        return
    parts = [f"[CALL][session_id={session_id}][stage={stage}][event={event}]"]
    if duration is not None:
        parts.append(f"[duration_ms={duration}]")
    if details:
        parts.extend(f"[{key}={_safe_value(value)}]" for key, value in details.items())
    LOGGER.log(level, "".join(parts))


def call_exception(
    session_id: str,
    stage: str,
    event: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Log an exception with the active traceback using the standard correlation fields."""
    parts = [f"[CALL][session_id={session_id}][stage={stage}][event={event}]"]
    if details:
        parts.extend(f"[{key}={_safe_value(value)}]" for key, value in details.items())
    LOGGER.exception("".join(parts))
