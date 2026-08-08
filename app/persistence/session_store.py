"""Persist every node transition as a redacted JSON-reconstructable snapshot."""

import json
from pathlib import Path
from typing import Any, cast

from sqlmodel import Session, SQLModel, create_engine, select

from app.config import settings
from app.graph.state import (
    ConversationState,
    EscalationDecision,
    GroundingResult,
    IntentResult,
    Turn,
)
from app.persistence.models import SessionSnapshot
from app.security.redaction import redact_pii

_db_path = settings.database_url.replace("sqlite:///", "")
if _db_path and _db_path != ":memory:":
    Path(_db_path).parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SQLModel.metadata.create_all(engine)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    return value


def serialize_state(state: ConversationState) -> str:
    """Encode validated state to a redacted JSON document safe for persistent logs."""
    payload = {
        "session_id": state["session_id"],
        "turns": [item.model_dump(mode="json") for item in state["turns"]],
        "current_transcript": state["current_transcript"],
        "intent_result": state["intent_result"].model_dump(mode="json") if state["intent_result"] else None,
        "rolling_sentiment": state["rolling_sentiment"],
        "clarification_count": state["clarification_count"],
        "draft_answer": state["draft_answer"],
        "retrieved_excerpts": state["retrieved_excerpts"],
        "grounding_result": state["grounding_result"].model_dump(mode="json") if state["grounding_result"] else None,
        "escalation_decision": (
            state["escalation_decision"].model_dump(mode="json") if state["escalation_decision"] else None
        ),
        "final_response_text": state["final_response_text"],
    }
    return json.dumps(_redact(payload), sort_keys=True)


def record_transition(node_name: str, state: ConversationState) -> None:
    """Write a redacted state snapshot after each graph node transition."""
    snapshot = SessionSnapshot(session_id=state["session_id"], node_name=node_name, state_json=serialize_state(state))
    with Session(engine) as session:
        session.add(snapshot)
        session.commit()


def load_latest_state(session_id: str) -> ConversationState | None:
    """Reconstruct the latest persisted state for a human-handoff payload."""
    statement = (
        select(SessionSnapshot).where(SessionSnapshot.session_id == session_id).order_by(SessionSnapshot.id.desc())
    )
    with Session(engine) as session:
        snapshot = session.exec(statement).first()
    if snapshot is None:
        return None
    data = json.loads(snapshot.state_json)
    data["turns"] = [Turn.model_validate(turn) for turn in data["turns"]]
    for key, model in (
        ("intent_result", IntentResult),
        ("grounding_result", GroundingResult),
        ("escalation_decision", EscalationDecision),
    ):
        if data.get(key) is not None:
            data[key] = model.model_validate(data[key])
    return cast(ConversationState, data)
