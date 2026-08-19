"""Regression coverage for transcript, TTS, and WebSocket failure boundaries."""

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from app.audio import gateway, tts
from app.audio.stt import StreamingTranscriber
from app.config import Settings

from .helpers import state


class FakeWebSocket:
    """Small in-memory WebSocket harness for the gateway's persistent receive loop."""

    def __init__(self) -> None:
        self.headers = {"origin": "http://localhost:3000"}
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.json_events: list[dict[str, Any]] = []
        self.bytes_events: list[bytes] = []
        self.closed = False

    async def accept(self) -> None:
        return None

    async def receive_json(self) -> dict[str, str]:
        return {"type": "auth", "token": "test-token"}

    async def receive(self) -> dict[str, Any]:
        return await self.events.get()

    async def send_json(self, event: dict[str, Any]) -> None:
        self.json_events.append(event)

    async def send_bytes(self, chunk: bytes) -> None:
        self.bytes_events.append(chunk)

    async def close(self, code: int) -> None:
        self.closed = True


class FakeSynthesizer:
    output_format = "audio/wav"

    def __init__(self, *, fail: bool = False, chunks: list[bytes] | None = None) -> None:
        self.fail = fail
        self.chunks = chunks or [b"RIFFtest"]
        self.interrupted = False

    def interrupt(self) -> None:
        self.interrupted = True

    async def stream(self, _text: str, _session_id: str) -> AsyncIterator[bytes]:
        if self.fail:
            raise ValueError("provider unavailable")
        for chunk in self.chunks:
            if self.interrupted:
                return
            yield chunk


class FakeGraph:
    def __init__(self, *, failure: bool = False) -> None:
        self.calls = 0
        self.failure = failure

    def invoke(self, input_state: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        if self.failure:
            raise RuntimeError("graph failed")
        return {
            **input_state,
            "final_response_text": "A grounded answer.",
            "rolling_sentiment": 0.2,
            "system_failure": None,
            "escalation_decision": None,
        }


async def _wait_for(events: list[dict[str, Any]], event_type: str) -> None:
    for _ in range(100):
        if any(event.get("type") == event_type for event in events):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {event_type}: {events}")


async def _run_typed_turn(
    monkeypatch: pytest.MonkeyPatch, *, graph: FakeGraph, synthesizer: FakeSynthesizer
) -> FakeWebSocket:
    socket = FakeWebSocket()
    monkeypatch.setattr(gateway.settings, "allowed_origins", ["http://localhost:3000"])
    monkeypatch.setattr(gateway, "verify_call_token", lambda *_: {})
    monkeypatch.setattr(gateway, "GRAPH", graph)
    monkeypatch.setattr(gateway, "InterruptibleSynthesizer", lambda: synthesizer)
    monkeypatch.setattr(gateway, "record_transition_safely", lambda *_: True)
    task = asyncio.create_task(gateway.serve_audio_socket(socket, "test-session"))
    await socket.events.put(
        {
            "type": "websocket.receive",
            "text": json.dumps({"type": "transcript", "text": "Where is my order?"}),
        }
    )
    await _wait_for(socket.json_events, "response" if not graph.failure else "backend_error")
    if not graph.failure:
        await _wait_for(socket.json_events, "tts_unavailable" if synthesizer.fail else "tts_complete")
    await socket.events.put({"type": "websocket.disconnect"})
    await asyncio.wait_for(task, timeout=1)
    return socket


def test_typed_transcript_reaches_graph_and_sends_text_before_tts_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = FakeGraph()
    socket = asyncio.run(_run_typed_turn(monkeypatch, graph=graph, synthesizer=FakeSynthesizer(fail=True)))
    event_types = [event["type"] for event in socket.json_events]
    assert graph.calls == 1
    assert event_types.index("response") < event_types.index("tts_unavailable")
    assert next(event for event in socket.json_events if event["type"] == "response")["text"] == "A grounded answer."


def test_graph_exception_sends_safe_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = asyncio.run(_run_typed_turn(monkeypatch, graph=FakeGraph(failure=True), synthesizer=FakeSynthesizer()))
    error = next(event for event in socket.json_events if event["type"] == "backend_error")
    assert error["stage"] == "graph"
    assert "RuntimeError" not in error["message"]


def test_malformed_websocket_event_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> FakeWebSocket:
        socket = FakeWebSocket()
        monkeypatch.setattr(gateway.settings, "allowed_origins", ["http://localhost:3000"])
        monkeypatch.setattr(gateway, "verify_call_token", lambda *_: {})
        monkeypatch.setattr(gateway, "record_transition_safely", lambda *_: True)
        task = asyncio.create_task(gateway.serve_audio_socket(socket, "test-session"))
        await socket.events.put({"type": "websocket.receive", "text": "not-json"})
        await _wait_for(socket.json_events, "input_rejected")
        await socket.events.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(task, timeout=1)
        return socket

    socket = asyncio.run(run())
    assert socket.json_events[-1]["message"] == "Expected a JSON event."


def test_empty_stt_result_is_handled_without_loading_a_model() -> None:
    assert StreamingTranscriber("test-session").transcribe_final() == ""


def test_near_duplicate_stt_finals_are_identified_but_corrections_are_preserved() -> None:
    duplicate, similarity = gateway._is_near_duplicate_transcript(
        "ello, I have some trouble getting through my payments",
        "Hello, I have some trouble getting through my payments",
    )
    assert duplicate
    assert similarity >= 0.96
    correction, _ = gateway._is_near_duplicate_transcript(
        "Hello, I have some trouble getting through my image",
        "Hello, I have some trouble getting through my payments",
    )
    assert not correction


def test_tts_complete_follows_all_audio_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[list[dict[str, Any]], list[bytes]]:
        events: list[dict[str, Any]] = []
        chunks: list[bytes] = []

        async def send_json(event: dict[str, Any]) -> bool:
            events.append(event)
            return True

        async def send_bytes(chunk: bytes) -> bool:
            chunks.append(chunk)
            return True

        monkeypatch.setattr(gateway, "record_transition_safely", lambda *_: True)
        await gateway._send_tts(
            send_json,
            send_bytes,
            FakeSynthesizer(chunks=[b"a", b"b"]),
            "answer",
            "test",
            lambda: state(),
        )
        return events, chunks

    events, chunks = asyncio.run(run())
    assert chunks == [b"a", b"b"]
    assert events[-1] == {"type": "tts_complete"}


def test_disconnect_during_tts_stops_without_claiming_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        async def send_json(event: dict[str, Any]) -> bool:
            events.append(event)
            return True

        async def send_bytes(_chunk: bytes) -> bool:
            return False

        monkeypatch.setattr(gateway, "record_transition_safely", lambda *_: True)
        await gateway._send_tts(
            send_json,
            send_bytes,
            FakeSynthesizer(chunks=[b"a"]),
            "answer",
            "test",
            lambda: state(),
        )
        return events

    assert {"type": "tts_complete"} not in asyncio.run(run())


def test_coqui_provider_produces_wav_and_honors_barge_in(monkeypatch: pytest.MonkeyPatch) -> None:
    class Engine:
        synthesizer = SimpleNamespace(output_sample_rate=16_000)

        def tts(self, text: str) -> np.ndarray:
            assert text == "hello"
            return np.zeros(8_000, dtype=np.float32)

    async def run() -> list[bytes]:
        synthesizer = tts.InterruptibleSynthesizer()
        synthesizer._coqui_engine = Engine()  # type: ignore[assignment]
        monkeypatch.setattr(tts.settings, "tts_provider", "coqui")
        monkeypatch.setattr(tts, "CHUNK_SIZE", 16)
        stream = synthesizer.stream("hello", "test")
        first = await anext(stream)
        synthesizer.interrupt()
        remainder = [chunk async for chunk in stream]
        return [first, *remainder]

    chunks = asyncio.run(run())
    assert chunks[0].startswith(b"RIFF")
    assert len(chunks) == 1


def test_elevenlabs_provider_uses_browser_decodable_mp3(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"
            self.text_to_speech = SimpleNamespace(convert=self.convert)

        def convert(self, **kwargs: Any) -> list[bytes]:
            assert kwargs["output_format"] == "mp3_44100_128"
            assert kwargs["voice_id"] == "voice-id"
            return [b"ID3", b"fake-mp3"]

    async def run() -> bytes:
        monkeypatch.setattr(tts.settings, "tts_provider", "elevenlabs")
        monkeypatch.setattr(tts.settings, "elevenlabs_api_key", "test-key")
        monkeypatch.setattr(tts.settings, "elevenlabs_voice_id", "voice-id")
        monkeypatch.setattr(tts, "ElevenLabs", Client)
        synthesizer = tts.InterruptibleSynthesizer()
        return b"".join([chunk async for chunk in synthesizer.stream("hello", "test")])

    assert asyncio.run(run()) == b"ID3fake-mp3"


def test_elevenlabs_missing_api_key_is_a_configuration_error() -> None:
    with pytest.raises(ValidationError, match="ELEVENLABS_API_KEY"):
        Settings(ollama_api_key="test", api_auth_secret="test", tts_provider="elevenlabs", elevenlabs_api_key=None)
