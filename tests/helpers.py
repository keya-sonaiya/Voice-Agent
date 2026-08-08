from datetime import UTC, datetime

from app.graph.state import ConversationState, Turn


def state(transcript: str = "Where is my order?") -> ConversationState:
    return {
        "session_id": "test-session",
        "turns": [Turn(role="caller", text=transcript, timestamp=datetime.now(UTC))],
        "current_transcript": transcript,
        "intent_result": None,
        "rolling_sentiment": 0.0,
        "clarification_count": 0,
        "draft_answer": None,
        "retrieved_excerpts": [],
        "grounding_result": None,
        "escalation_decision": None,
        "final_response_text": None,
    }
