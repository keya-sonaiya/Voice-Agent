"""Build secret-free escalation contexts from persisted state only."""

from typing import Any

from app.persistence.session_store import load_latest_state


def build_handoff_payload(session_id: str) -> dict[str, Any]:
    """Build a human-readable escalation payload from the latest persisted snapshot."""
    state = load_latest_state(session_id)
    if state is None:
        raise LookupError("No persisted session state exists for this handoff.")
    return {
        "session_id": state["session_id"],
        "transcript": [
            {"role": turn.role, "text": turn.text, "timestamp": turn.timestamp.isoformat()} for turn in state["turns"]
        ],
        "intent": state["intent_result"].model_dump() if state["intent_result"] else None,
        "rolling_sentiment": state["rolling_sentiment"],
        "escalation": (state["escalation_decision"].model_dump() if state["escalation_decision"] else None),
        "attempted_answer": state["draft_answer"],
        "grounding": state["grounding_result"].model_dump() if state["grounding_result"] else None,
    }
