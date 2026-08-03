"""Deterministic escalation-rule node."""

from app.config import settings
from app.graph.state import ConversationState, EscalationDecision


def decide_escalation(state: ConversationState) -> dict[str, EscalationDecision]:
    """Read intent, grounding, sentiment, and clarifications; write `escalation_decision`; never call an LLM."""
    intent = state["intent_result"]
    grounding = state["grounding_result"]
    if grounding is not None and not grounding.is_grounded:
        decision = EscalationDecision(should_escalate=True, reason="grounding_failure")
    elif intent is not None and intent.confidence < settings.confidence_threshold:
        decision = EscalationDecision(should_escalate=True, reason="low_confidence")
    elif state["rolling_sentiment"] < settings.sentiment_escalation_threshold:
        decision = EscalationDecision(should_escalate=True, reason="negative_sentiment_trend")
    elif state["clarification_count"] > settings.max_clarifications:
        decision = EscalationDecision(should_escalate=True, reason="repeated_clarification")
    elif intent is not None and intent.intent == "human_request":
        decision = EscalationDecision(should_escalate=True, reason="explicit_human_request")
    else:
        decision = EscalationDecision(should_escalate=False, reason="none")
    return {"escalation_decision": decision}
