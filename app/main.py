"""FastAPI entrypoint and protected WebSocket routes."""

import hmac
import logging
import uuid

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.audio.gateway import serve_audio_socket
from app.config import settings
from app.escalation.handoff import build_handoff_payload
from app.security.auth import issue_call_token
from app.security.rate_limit import enforce_rate_limit

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="Voice-Driven Customer Support Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


class CallStart(BaseModel):
    caller_id: str


@app.get("/health")
def health() -> dict[str, str]:
    """Return a no-secret liveness signal."""
    return {"status": "ok"}


@app.post("/calls")
def start_call(body: CallStart, request: Request) -> dict[str, str]:
    """Rate-limit and issue a short-lived token for an authenticated call session."""
    client_ip = request.client.host if request.client else "unknown"
    if not enforce_rate_limit(f"ip:{client_ip}") or not enforce_rate_limit(f"caller:{body.caller_id}"):
        raise HTTPException(status_code=429, detail="Call initiation limit exceeded.")
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "token": issue_call_token(session_id, body.caller_id)}


@app.websocket("/ws/audio/{session_id}")
async def audio_gateway(websocket: WebSocket, session_id: str) -> None:
    """Serve the persistent authenticated duplex audio connection."""
    await serve_audio_socket(websocket, session_id)


@app.get("/internal/handoff/{session_id}")
def handoff(session_id: str, authorization: str = Header(default="")) -> dict[str, object]:
    """Expose an internal human-queue adapter protected by service-to-service auth."""
    expected = f"Bearer {settings.api_auth_secret}"
    if not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Service authorization required.")
    return build_handoff_payload(session_id)
