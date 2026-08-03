"""Short-lived token helpers for the audio WebSocket."""

from datetime import UTC, datetime, timedelta
from typing import cast

import jwt
from jwt import InvalidTokenError

from app.config import settings


class AuthenticationError(ValueError):
    """Raised when a caller token is missing, invalid, or expired."""


def issue_call_token(session_id: str, caller_id: str) -> str:
    """Create a short-lived signed token scoped to one caller session."""
    payload = {
        "session_id": session_id,
        "caller_id": caller_id,
        "exp": datetime.now(UTC) + timedelta(minutes=15),
    }
    return jwt.encode(payload, settings.api_auth_secret, algorithm="HS256")


def verify_call_token(token: str, session_id: str) -> dict[str, object]:
    """Validate signature, expiration, and session binding for a WS caller."""
    try:
        payload = jwt.decode(token, settings.api_auth_secret, algorithms=["HS256"])
    except InvalidTokenError as error:
        raise AuthenticationError("Invalid call token.") from error
    if payload.get("session_id") != session_id:
        raise AuthenticationError("Call token is not valid for this session.")
    return cast(dict[str, object], payload)
