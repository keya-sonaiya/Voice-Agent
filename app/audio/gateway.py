"""Authenticated duplex WebSocket gateway for transcription and speech playback."""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, MutableMapping
from datetime import UTC, datetime
from difflib import SequenceMatcher
from time import monotonic
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.audio.stt import StreamingTranscriber
from app.audio.tts import InterruptibleSynthesizer
from app.call_logging import call_exception, call_log, duration_ms
from app.config import settings
from app.escalation.handoff import build_handoff_payload
from app.graph.build_graph import build_graph
from app.graph.state import ConversationState, Turn
from app.persistence.session_store import record_transition_safely
from app.security.auth import AuthenticationError, verify_call_token
from app.security.input_guard import UnsafeTranscriptError, validate_transcript
from app.security.rate_limit import enforce_rate_limit

GRAPH = build_graph()
logger = logging.getLogger(__name__)

SendJson = Callable[[dict[str, Any]], Awaitable[bool]]
SendBytes = Callable[[bytes], Awaitable[bool]]
_DUPLICATE_TRANSCRIPT_WINDOW_SECONDS = 10.0


def _normalise_transcript_for_deduplication(transcript: str) -> str:
    """Compare transcript events without treating casing or punctuation as distinct turns."""
    return " ".join(re.findall(r"[a-z0-9]+", transcript.lower()))


def _is_near_duplicate_transcript(previous: str, current: str) -> tuple[bool, float]:
    """Identify repeated STT finals while preserving meaningfully different caller corrections."""
    previous_normalised = _normalise_transcript_for_deduplication(previous)
    current_normalised = _normalise_transcript_for_deduplication(current)
    if not previous_normalised or not current_normalised:
        return False, 0.0
    similarity = SequenceMatcher(None, previous_normalised, current_normalised).ratio()
    return similarity >= 0.96, similarity


def initial_state(session_id: str, authenticated_caller_id: str | None = None) -> ConversationState:
    """Create the complete typed state required for a fresh caller session."""
    return {
        "session_id": session_id,
        "authenticated_caller_id": authenticated_caller_id,
        "turns": [],
        "current_transcript": "",
        "intent_result": None,
        "previous_intent": None,
        "rolling_sentiment": 0.0,
        "clarification_count": 0,
        "awaiting_clarification": False,
        "clarification_topic": None,
        "clarification_resolved": False,
        "customer_id": None,
        "customer_identified": False,
        "customer_verified": False,
        "identity_state": "unidentified",
        "verification_method": None,
        "verification_timestamp": None,
        "awaiting_customer_name": False,
        "awaiting_customer_phone": False,
        "awaiting_customer_email": False,
        "account_recovery_active": False,
        "account_recovery_attempts": 0,
        "recovery_candidate_ids": [],
        "support_intent": None,
        "current_payment_id": None,
        "current_invoice_id": None,
        "current_order_id": None,
        "current_ticket_id": None,
        "awaiting_customer_verification": False,
        "awaiting_payment_id": False,
        "draft_answer": None,
        "retrieved_excerpts": [],
        "grounding_result": None,
        "escalation_decision": None,
        "final_response_text": None,
        "system_failure": None,
        "response_mode": None,
    }


async def _send_tts(
    send_json: SendJson,
    send_bytes: SendBytes,
    synthesizer: InterruptibleSynthesizer,
    text: str,
    session_id: str,
    current_state: Callable[[], ConversationState],
) -> None:
    """Send complete WAV/MP3 bytes in order, while surfacing every TTS task outcome."""
    started = monotonic()
    chunk_count = 0
    byte_count = 0
    record_transition_safely("tts_started", current_state())
    call_log(
        session_id,
        "TTS",
        "start",
        details={"provider": settings.tts_provider, "text_length": len(text), "format": synthesizer.output_format},
    )
    if not await send_json({"type": "tts_started", "format": synthesizer.output_format}):
        return
    try:
        async for chunk in synthesizer.stream(text, session_id):
            if not chunk:
                continue
            if not await send_bytes(chunk):
                return
            chunk_count += 1
            byte_count += len(chunk)
            call_log(
                session_id,
                "TTS",
                "chunk_sent",
                details={"chunk_count": chunk_count, "sent_bytes": byte_count},
            )
        record_transition_safely("tts_complete", current_state())
        call_log(
            session_id,
            "TTS",
            "complete",
            duration=duration_ms(started),
            details={"chunk_count": chunk_count, "output_bytes": byte_count},
        )
        await send_json({"type": "tts_complete"})
    except asyncio.CancelledError:
        record_transition_safely("tts_interrupted", current_state())
        call_log(session_id, "TTS", "interrupted", details={"chunk_count": chunk_count})
        await send_json({"type": "tts_interrupted"})
        raise
    except Exception:
        record_transition_safely("tts_failed", current_state())
        call_exception(
            session_id,
            "TTS",
            "failed",
            details={"provider": settings.tts_provider, "chunk_count": chunk_count},
        )
        await send_json({"type": "tts_unavailable", "message": "Speech playback is temporarily unavailable."})


async def serve_audio_socket(websocket: WebSocket, session_id: str) -> None:
    """Run one authenticated audio call; all audio and speech share this WS connection."""
    origin = websocket.headers.get("origin")
    if origin not in settings.allowed_origins:
        call_log(session_id, "AUTH", "origin_rejected", level=logging.WARNING, details={"origin": origin or "missing"})
        await websocket.close(code=1008)
        return
    await websocket.accept()
    call_log(session_id, "WS", "connected")
    try:
        first_message = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        token = first_message.get("token") if first_message.get("type") == "auth" else None
        if not isinstance(token, str):
            raise AuthenticationError("WebSocket authentication is required.")
        claims = verify_call_token(token, session_id)
    except (AuthenticationError, asyncio.TimeoutError, ValueError, TypeError):
        call_log(session_id, "AUTH", "failed", level=logging.WARNING)
        await websocket.close(code=1008)
        return
    call_log(session_id, "AUTH", "success")

    caller_claim = claims.get("caller_id")
    authenticated_caller_id = caller_claim if isinstance(caller_claim, str) else None
    state = initial_state(session_id, authenticated_caller_id)
    call_log(session_id, "IDENTITY", "state", details={"identity_state": state["identity_state"]})
    record_transition_safely("session_started", state)
    started = monotonic()
    transcriber = StreamingTranscriber(session_id)
    synthesizer = InterruptibleSynthesizer()
    tts_task: asyncio.Task[None] | None = None
    transcript_tasks: set[asyncio.Task[None]] = set()
    process_lock = asyncio.Lock()
    send_lock = asyncio.Lock()
    last_accepted_transcript: tuple[str, float] | None = None

    async def send_json(event: dict[str, Any]) -> bool:
        """Serialise outbound JSON with binary chunks and convert disconnects into a clean stop."""
        try:
            async with send_lock:
                await websocket.send_json(event)
        except (RuntimeError, WebSocketDisconnect):
            call_log(session_id, "WS", "send_failed", level=logging.WARNING, details={"event_type": event.get("type")})
            return False
        event_type = event.get("type")
        if event_type == "response":
            call_log(session_id, "WS", "response_sent", details={"response_length": len(str(event.get("text", "")))})
        elif event_type in {"backend_error", "tts_unavailable", "escalation"}:
            call_log(session_id, "WS", "event_sent", details={"event_type": event_type})
        return True

    async def send_bytes(chunk: bytes) -> bool:
        try:
            async with send_lock:
                await websocket.send_bytes(chunk)
            return True
        except (RuntimeError, WebSocketDisconnect):
            call_log(session_id, "WS", "audio_send_failed", level=logging.WARNING, details={"chunk_bytes": len(chunk)})
            return False

    async def stop_active_tts() -> None:
        """Cancel current playback so a caller can barge in without stale audio leaking."""
        nonlocal tts_task
        synthesizer.interrupt()
        if tts_task is not None and not tts_task.done():
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass
        tts_task = None

    async def send_escalation(reason: str) -> None:
        """Send a usable escalation event even when persisted-handoff reconstruction fails."""
        try:
            payload = build_handoff_payload(session_id)
            call_log(session_id, "HANDOFF", "payload_built", details={"reason": reason})
        except Exception:
            call_exception(session_id, "HANDOFF", "payload_failed", details={"reason": reason})
            payload = {
                "session_id": session_id,
                "transcript": [],
                "escalation": {"should_escalate": True, "reason": reason},
                "attempted_answer": None,
            }
        await send_json({"type": "escalation", "payload": payload})

    async def process_transcript(transcript: str, received_at: float) -> None:
        """Validate and run one text turn; every unexpected failure reaches the frontend safely."""
        nonlocal last_accepted_transcript, state, tts_task
        try:
            call_log(session_id, "TRANSCRIPT", "received", details={"transcript_length": len(transcript)})
            try:
                transcript = validate_transcript(transcript)
            except UnsafeTranscriptError as error:
                call_log(session_id, "TRANSCRIPT", "rejected", level=logging.WARNING)
                await send_json({"type": "input_rejected", "message": str(error)})
                return
            call_log(session_id, "TRANSCRIPT", "validated", details={"transcript_length": len(transcript)})

            async with process_lock:
                if last_accepted_transcript is not None:
                    previous_transcript, previous_received_at = last_accepted_transcript
                    is_duplicate, similarity = _is_near_duplicate_transcript(previous_transcript, transcript)
                    if is_duplicate and received_at - previous_received_at <= _DUPLICATE_TRANSCRIPT_WINDOW_SECONDS:
                        call_log(
                            session_id,
                            "TRANSCRIPT",
                            "duplicate_ignored",
                            details={"similarity": round(similarity, 3)},
                        )
                        return
                if not enforce_rate_limit(f"session:{session_id}"):
                    call_log(session_id, "TRANSCRIPT", "rate_limited", level=logging.WARNING)
                    await send_json({"type": "input_rejected", "message": "Transcript rate limit exceeded."})
                    return
                last_accepted_transcript = (transcript, received_at)
                caller_turn = Turn(role="caller", text=transcript, timestamp=datetime.now(UTC))
                state = {
                    **state,
                    "current_transcript": transcript,
                    "previous_intent": state["intent_result"],
                    "clarification_resolved": False,
                    "turns": [*state["turns"], caller_turn],
                }
                record_transition_safely("transcript_received", state)
                await send_json({"type": "transcript", "text": transcript, "final": True})

                graph_started = monotonic()
                call_log(session_id, "GRAPH", "start")
                try:
                    state = await asyncio.to_thread(GRAPH.invoke, state)
                except Exception:
                    call_exception(session_id, "GRAPH", "failed")
                    await send_json(
                        {
                            "type": "backend_error",
                            "stage": "graph",
                            "message": "The response pipeline failed. Check backend logs.",
                        }
                    )
                    return
                call_log(
                    session_id,
                    "GRAPH",
                    "complete",
                    duration=duration_ms(graph_started),
                    details={"system_failure": state["system_failure"] or "none"},
                )
                if state["system_failure"]:
                    call_log(
                        session_id,
                        "GRAPH",
                        "ended_without_response",
                        level=logging.ERROR,
                        details={"reason": state["system_failure"]},
                    )
                    await send_json(
                        {
                            "type": "backend_error",
                            "stage": state["system_failure"],
                            "message": "The response pipeline failed. Check backend logs.",
                        }
                    )
                    return
                decision = state["escalation_decision"]
                if decision is not None and decision.should_escalate:
                    call_log(
                        session_id,
                        "GRAPH",
                        "ended_without_response",
                        details={"reason": decision.reason},
                    )
                    call_log(session_id, "ESCALATION", "selected", details={"reason": decision.reason})
                    await send_escalation(decision.reason)
                    return
                response = state["final_response_text"]
                if not isinstance(response, str) or not response.strip():
                    call_log(
                        session_id,
                        "GRAPH",
                        "ended_without_response",
                        level=logging.ERROR,
                        details={"reason": "missing_final_response_text"},
                    )
                    await send_json(
                        {
                            "type": "backend_error",
                            "stage": "response",
                            "message": "The response pipeline failed. Check backend logs.",
                        }
                    )
                    return
                agent_turn = Turn(role="agent", text=response, timestamp=datetime.now(UTC))
                state = {**state, "turns": [*state["turns"], agent_turn]}
                record_transition_safely("respond", state)
                await stop_active_tts()
                # Text is sent before synthesis starts: speech outages cannot hide the answer.
                if not await send_json({"type": "response", "text": response, "sentiment": state["rolling_sentiment"]}):
                    return
                record_transition_safely("tts_queued", state)
                tts_task = asyncio.create_task(
                    _send_tts(send_json, send_bytes, synthesizer, response, session_id, lambda: state),
                    name=f"tts:{session_id}",
                )
        except asyncio.CancelledError:
            call_log(session_id, "GRAPH", "cancelled")
            raise
        except Exception:
            call_exception(session_id, "GRAPH", "unhandled_task_failure")
            await send_json(
                {
                    "type": "backend_error",
                    "stage": "graph",
                    "message": "The response pipeline failed. Check backend logs.",
                }
            )

    def queue_transcript(transcript: str) -> None:
        """Track transcript tasks so no background exception can disappear unnoticed."""
        task = asyncio.create_task(process_transcript(transcript, monotonic()), name=f"transcript:{session_id}")
        transcript_tasks.add(task)

        def completed(completed_task: asyncio.Task[None]) -> None:
            transcript_tasks.discard(completed_task)
            try:
                completed_task.result()
            except asyncio.CancelledError:
                return
            except Exception as error:  # Defensive guard for failures outside process_transcript's boundary.
                logger.error(
                    "[CALL][session_id=%s][stage=GRAPH][event=background_task_failed]",
                    session_id,
                    exc_info=(type(error), error, error.__traceback__),
                )

        task.add_done_callback(completed)

    try:
        while True:
            remaining = settings.max_call_duration_seconds - (monotonic() - started)
            if remaining <= 0:
                await send_json({"type": "call_terminated", "message": "Maximum call duration reached."})
                await websocket.close(code=1008)
                return
            try:
                message: MutableMapping[str, Any] = await asyncio.wait_for(websocket.receive(), timeout=remaining)
            except asyncio.TimeoutError:
                await send_json({"type": "call_terminated", "message": "Maximum call duration reached."})
                await websocket.close(code=1008)
                return
            if message.get("type") == "websocket.disconnect":
                call_log(session_id, "WS", "disconnected")
                return
            audio = message.get("bytes")
            if isinstance(audio, bytes):
                transcriber.add_audio(audio)
                call_log(
                    session_id,
                    "AUDIO",
                    "chunk_received",
                    details={
                        "chunk_bytes": len(audio),
                        "frame_count": transcriber.audio_frame_count,
                        "accumulated_bytes": transcriber.audio_byte_count,
                    },
                )
                if transcriber.contains_speech(audio):
                    await stop_active_tts()
                continue
            text = message.get("text")
            if text is None:
                call_log(session_id, "WS", "ignored_frame", level=logging.WARNING)
                continue
            try:
                event = json.loads(text)
            except (TypeError, ValueError):
                call_log(session_id, "WS", "malformed_event", level=logging.WARNING)
                await send_json({"type": "input_rejected", "message": "Expected a JSON event."})
                continue
            if not isinstance(event, dict):
                call_log(session_id, "WS", "invalid_event", level=logging.WARNING)
                await send_json({"type": "input_rejected", "message": "Expected a JSON event object."})
                continue
            event_type = event.get("type")
            if event_type == "audio_end":
                call_log(
                    session_id,
                    "STT",
                    "audio_end",
                    details={"frame_count": transcriber.audio_frame_count, "audio_bytes": transcriber.audio_byte_count},
                )
                try:
                    transcript = await asyncio.to_thread(transcriber.transcribe_final)
                except Exception:
                    call_exception(session_id, "STT", "failed")
                    await send_json(
                        {
                            "type": "backend_error",
                            "stage": "stt",
                            "message": "Speech recognition failed. Check backend logs.",
                        }
                    )
                    continue
                if transcript:
                    queue_transcript(transcript)
                else:
                    await send_json({"type": "transcript_empty", "message": "No speech was detected."})
            elif event_type == "transcript":
                await stop_active_tts()
                typed_transcript = event.get("text")
                if isinstance(typed_transcript, str):
                    call_log(
                        session_id,
                        "WS",
                        "transcript_event_received",
                        details={"text_length": len(typed_transcript)},
                    )
                    queue_transcript(typed_transcript)
                else:
                    await send_json({"type": "input_rejected", "message": "Transcript text is required."})
            else:
                call_log(
                    session_id,
                    "WS",
                    "unsupported_event",
                    level=logging.WARNING,
                    details={"event_type": event_type},
                )
                await send_json({"type": "input_rejected", "message": "Unsupported WebSocket event."})
    except WebSocketDisconnect:
        call_log(session_id, "WS", "disconnected")
    except Exception:
        call_exception(session_id, "WS", "gateway_failed")
        await send_json(
            {"type": "backend_error", "stage": "gateway", "message": "The call connection failed. Check backend logs."}
        )
    finally:
        for task in transcript_tasks:
            task.cancel()
        if transcript_tasks:
            await asyncio.gather(*transcript_tasks, return_exceptions=True)
        await stop_active_tts()
