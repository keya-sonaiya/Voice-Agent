"""Persist every node transition as a redacted JSON-reconstructable snapshot."""

import json
from pathlib import Path
from typing import Any, cast

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine, select

from app.call_logging import call_exception, call_log
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


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection: Any, _: object) -> None:
    """Make SQLite enforce the ownership/integrity relationships used by support tools."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


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
        "authenticated_caller_id": state["authenticated_caller_id"],
        "turns": [item.model_dump(mode="json") for item in state["turns"]],
        "current_transcript": state["current_transcript"],
        "intent_result": state["intent_result"].model_dump(mode="json") if state["intent_result"] else None,
        "previous_intent": state["previous_intent"].model_dump(mode="json") if state["previous_intent"] else None,
        "rolling_sentiment": state["rolling_sentiment"],
        "clarification_count": state["clarification_count"],
        "awaiting_clarification": state["awaiting_clarification"],
        "clarification_topic": state["clarification_topic"],
        "clarification_resolved": state["clarification_resolved"],
        "customer_id": state["customer_id"],
        "customer_identified": state["customer_identified"],
        "customer_verified": state["customer_verified"],
        "identity_state": state["identity_state"],
        "verification_method": state["verification_method"],
        "verification_timestamp": state["verification_timestamp"],
        "awaiting_customer_name": state["awaiting_customer_name"],
        "awaiting_customer_phone": state["awaiting_customer_phone"],
        "awaiting_customer_email": state["awaiting_customer_email"],
        "account_recovery_active": state["account_recovery_active"],
        "account_recovery_attempts": state["account_recovery_attempts"],
        "recovery_candidate_ids": state["recovery_candidate_ids"],
        "support_intent": state["support_intent"],
        "current_payment_id": state["current_payment_id"],
        "current_invoice_id": state["current_invoice_id"],
        "current_order_id": state["current_order_id"],
        "current_ticket_id": state["current_ticket_id"],
        "awaiting_customer_verification": state["awaiting_customer_verification"],
        "awaiting_payment_id": state["awaiting_payment_id"],
        "draft_answer": state["draft_answer"],
        "retrieved_excerpts": state["retrieved_excerpts"],
        "grounding_result": state["grounding_result"].model_dump(mode="json") if state["grounding_result"] else None,
        "escalation_decision": (
            state["escalation_decision"].model_dump(mode="json") if state["escalation_decision"] else None
        ),
        "final_response_text": state["final_response_text"],
        "system_failure": state["system_failure"],
        "response_mode": state["response_mode"],
    }
    return json.dumps(_redact(payload), sort_keys=True)


def record_transition(node_name: str, state: ConversationState) -> None:
    """Write a redacted state snapshot after each graph node transition."""
    snapshot = SessionSnapshot(session_id=state["session_id"], node_name=node_name, state_json=serialize_state(state))
    with Session(engine) as session:
        session.add(snapshot)
        session.commit()


def record_transition_safely(node_name: str, state: ConversationState) -> bool:
    """Persist without allowing a database outage to end the caller's live turn."""
    try:
        record_transition(node_name, state)
    except Exception:
        call_exception(state["session_id"], "PERSISTENCE", "failed", details={"transition": node_name})
        return False
    call_log(state["session_id"], "PERSISTENCE", "complete", details={"transition": node_name})
    return True


def load_latest_state(session_id: str) -> ConversationState | None:
    """Reconstruct the latest persisted state for a human-handoff payload."""
    statement = (
        select(SessionSnapshot)
        .where(SessionSnapshot.session_id == session_id)
        .order_by(cast(Any, SessionSnapshot.id).desc())
    )
    with Session(engine) as session:
        snapshot = session.exec(statement).first()
    if snapshot is None:
        return None
    data = json.loads(snapshot.state_json)
    data.setdefault("system_failure", None)
    data.setdefault("response_mode", None)
    data.setdefault("authenticated_caller_id", None)
    data.setdefault("previous_intent", None)
    data.setdefault("awaiting_clarification", False)
    data.setdefault("clarification_topic", None)
    data.setdefault("clarification_resolved", False)
    data.setdefault("customer_id", None)
    data.setdefault("customer_identified", bool(data.get("customer_id")))
    data.setdefault("customer_verified", False)
    data.setdefault("identity_state", "verified" if data.get("customer_verified") else "identified" if data.get("customer_identified") else "unidentified")
    data.setdefault("verification_method", None)
    data.setdefault("verification_timestamp", None)
    data.setdefault("awaiting_customer_name", False)
    data.setdefault("awaiting_customer_phone", False)
    data.setdefault("awaiting_customer_email", False)
    data.setdefault("account_recovery_active", False)
    data.setdefault("account_recovery_attempts", 0)
    data.setdefault("recovery_candidate_ids", [])
    data.setdefault("support_intent", None)
    data.setdefault("current_payment_id", None)
    data.setdefault("current_invoice_id", None)
    data.setdefault("current_order_id", None)
    data.setdefault("current_ticket_id", None)
    data.setdefault("awaiting_customer_verification", False)
    data.setdefault("awaiting_payment_id", False)
    data["turns"] = [Turn.model_validate(turn) for turn in data["turns"]]
    for key, model in (
        ("intent_result", IntentResult),
        ("previous_intent", IntentResult),
        ("grounding_result", GroundingResult),
        ("escalation_decision", EscalationDecision),
    ):
        if data.get(key) is not None:
            data[key] = model.model_validate(data[key])
    return cast(ConversationState, data)
