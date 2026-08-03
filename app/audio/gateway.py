"""Authenticated duplex WebSocket gateway for transcription and speech playback."""

import asyncio
import json
from collections.abc import MutableMapping
from datetime import UTC, datetime
from time import monotonic
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.audio.stt import StreamingTranscriber
from app.audio.tts import InterruptibleSynthesizer
from app.config import settings
from app.escalation.handoff import build_handoff_payload
from app.graph.build_graph import build_graph
from app.graph.state import ConversationState, Turn
from app.persistence.session_store import record_transition
from app.security.auth import AuthenticationError, verify_call_token
from app.security.input_guard import UnsafeTranscriptError, validate_transcript

GRAPH = build_graph()


def initial_state(session_id: str) -> ConversationState:
    """Create the complete typed state required for a fresh caller session."""
    return {
        "session_id": session_id,
        "turns": [],
        "current_transcript": "",
        "intent_result": None,
        "rolling_sentiment": 0.0,
        "clarification_count": 0,
        "draft_answer": None,
        "grounding_result": None,
        "escalation_decision": None,
        "final_response_text": None,
    }


async def _send_tts(websocket: WebSocket, synthesizer: InterruptibleSynthesizer, text: str) -> None:
    """Send synthesized binary chunks over the existing duplex WebSocket."""
    try:
        async for chunk in synthesizer.stream(text):
            await websocket.send_bytes(chunk)
        await websocket.send_json({"type": "tts_complete"})
    except (OSError, RuntimeError):
        await websocket.send_json({"type": "tts_unavailable", "message": "Speech playback is temporarily unavailable."})


async def serve_audio_socket(websocket: WebSocket, session_id: str, token: str) -> None:
    """Run one authenticated audio call; all audio and speech share this WS connection."""
    try:
        verify_call_token(token, session_id)
    except AuthenticationError:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    state = initial_state(session_id)
    record_transition("session_started", state)
    started = monotonic()
    transcriber = StreamingTranscriber()
    synthesizer = InterruptibleSynthesizer()
    tts_task: asyncio.Task[None] | None = None
    process_lock = asyncio.Lock()

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

    async def process_transcript(transcript: str) -> None:
        nonlocal state, tts_task
        try:
            transcript = validate_transcript(transcript)
        except UnsafeTranscriptError as error:
            await websocket.send_json({"type": "input_rejected", "message": str(error)})
            return
        async with process_lock:
            caller_turn = Turn(role="caller", text=transcript, timestamp=datetime.now(UTC))
            state = {
                **state,
                "current_transcript": transcript,
                "turns": [*state["turns"], caller_turn],
            }
            record_transition("transcript_received", state)
            await websocket.send_json({"type": "transcript", "text": transcript, "final": True})
            state = await asyncio.to_thread(GRAPH.invoke, state)
            decision = state["escalation_decision"]
            if decision is not None and decision.should_escalate:
                await websocket.send_json({"type": "escalation", "payload": build_handoff_payload(session_id)})
                return
            response = state["final_response_text"]
            if response is None:
                await websocket.send_json({"type": "escalation", "payload": build_handoff_payload(session_id)})
                return
            agent_turn = Turn(role="agent", text=response, timestamp=datetime.now(UTC))
            state = {**state, "turns": [*state["turns"], agent_turn]}
            record_transition("tts_queued", state)
            await websocket.send_json({"type": "response", "text": response, "sentiment": state["rolling_sentiment"]})
            await stop_active_tts()
            tts_task = asyncio.create_task(_send_tts(websocket, synthesizer, response))

    try:
        while True:
            if monotonic() - started > settings.max_call_duration_seconds:
                await websocket.send_json({"type": "call_terminated", "message": "Maximum call duration reached."})
                await websocket.close(code=1008)
                return
            message: MutableMapping[str, Any] = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            audio = message.get("bytes")
            if isinstance(audio, bytes):
                await stop_active_tts()
                transcriber.add_audio(audio)
                continue
            text = message.get("text")
            if text is None:
                continue
            try:
                event = json.loads(text)
            except ValueError:
                await websocket.send_json({"type": "input_rejected", "message": "Expected a JSON event."})
                continue
            event_type = event.get("type")
            if event_type == "audio_end":
                transcript = await asyncio.to_thread(transcriber.transcribe_final)
                if transcript:
                    asyncio.create_task(process_transcript(transcript))
            elif event_type == "transcript":
                await stop_active_tts()
                transcript = event.get("text")
                if isinstance(transcript, str):
                    asyncio.create_task(process_transcript(transcript))
                else:
                    await websocket.send_json({"type": "input_rejected", "message": "Transcript text is required."})
    except WebSocketDisconnect:
        return
    finally:
        await stop_active_tts()
