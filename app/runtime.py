"""Application resource lifecycle and readiness state."""

from threading import Lock
from time import monotonic
from typing import Any
import logging

from app.call_logging import call_exception, call_log, duration_ms
from app.config import settings

_lock = Lock()
_status: dict[str, bool] = {
    "database": False,
    "stt": False,
    "embeddings": False,
    "vector_store": False,
    "tts": False,
    "llm": False,
}
APP_READY = False
_LOGGER = logging.getLogger("app.startup")


def _startup_log(stage: str, event: str, *, started: float | None = None, **details: object) -> None:
    parts = [f"[STARTUP][stage={stage}][event={event}]"]
    if started is not None:
        parts.append(f"[duration_ms={duration_ms(started)}]")
    parts.extend(f"[{key}={value}]" for key, value in details.items())
    _LOGGER.info("".join(parts))


def readiness() -> dict[str, Any]:
    with _lock:
        return {"ready": APP_READY, **_status}


def _set_status(component: str, value: bool) -> None:
    with _lock:
        _status[component] = value


def initialize_resources() -> None:
    """Warm mandatory local resources and reusable provider clients once per process."""
    global APP_READY
    started = monotonic()
    try:
        from app.persistence.session_store import engine
        from sqlmodel import SQLModel

        component_started = monotonic()
        _startup_log("DATABASE", "start")
        SQLModel.metadata.create_all(engine)
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        _set_status("database", True)
        _startup_log("DATABASE", "ready", started=component_started)

        from app.audio.stt import warm_whisper_model

        component_started = monotonic()
        _startup_log("STT", "start", model=settings.whisper_model)
        warm_whisper_model()
        _set_status("stt", True)
        _startup_log("STT", "ready", started=component_started)

        from app.rag.retriever import warm_retriever

        component_started = monotonic()
        _startup_log("EMBEDDINGS", "start")
        warm_retriever()
        _set_status("embeddings", True)
        _set_status("vector_store", True)
        _startup_log("EMBEDDINGS", "ready", started=component_started)
        _startup_log("VECTORSTORE", "ready")

        from app.graph.nodes import grounding_judge, intent_agent, knowledge_agent

        intent_agent.get_llm_client()
        knowledge_agent.get_llm_client()
        grounding_judge.get_llm_client()
        _set_status("llm", True)
        _startup_log("LLM", "ready")

        from app.audio.tts import warm_tts

        component_started = monotonic()
        _startup_log("TTS", "start", provider=settings.tts_provider)
        warm_tts()
        _set_status("tts", True)
        _startup_log("TTS", "ready", started=component_started)
    except Exception:
        call_exception("startup", "STARTUP", "failed")
        raise
    APP_READY = True
    _startup_log("APPLICATION", "ready")
    _startup_log("APPLICATION", "complete", started=started)