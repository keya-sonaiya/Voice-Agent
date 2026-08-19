from datetime import UTC, datetime

from app.graph.state import ConversationState, Turn


def state(transcript: str = "Where is my order?") -> ConversationState:
    return {
        "session_id": "test-session",
        "authenticated_caller_id": None,
        "turns": [Turn(role="caller", text=transcript, timestamp=datetime.now(UTC))],
        "current_transcript": transcript,
        "intent_result": None,
        "previous_intent": None,
        "rolling_sentiment": 0.0,
        "clarification_count": 0,
        "awaiting_clarification": False,
        "clarification_topic": None,
        "clarification_resolved": False,
        "customer_id": None,
        "customer_verified": False,
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
