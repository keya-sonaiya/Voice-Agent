"""Deterministic safe replies for greetings and first-pass clarification turns."""

import re

from app.call_logging import call_log
from app.graph.state import ConversationState

_GREETING_WORDS = {"hello", "hi", "hey", "howdy", "greetings"}
_GREETING_PHRASES = {"good morning", "good afternoon", "good evening"}
_HELP_PHRASES = {
    "can you help me",
    "can you help",
    "could you help me",
    "what can you do",
    "what do you do",
    "i need help",
}
_HUMAN_REQUEST_PATTERN = re.compile(r"\b(?:human|live agent|real person|representative|supervisor)\b", re.IGNORECASE)

_GREETING_RESPONSE = "Hello! How can I help you today?"
_INTRODUCTION_RESPONSE = "Nice to meet you. How can I help you today?"
_CAPABILITY_RESPONSE = (
    "I can help with order status, billing, cancellations, account access, and technical issues. "
    "What do you need help with?"
)
_CLARIFICATION_RESPONSE = (
    "Sure. Could you tell me whether you need help with your order, payment, cancellation, account, or something else?"
)
_FOLLOW_UP_CLARIFICATION_RESPONSE = (
    "I still need a little more detail. Is this about a payment, order status, cancellation, account access, "
    "or a technical issue?"
)


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def is_explicit_human_request(text: str) -> bool:
    """Recognise a direct request for a person even if the intent model is uncertain."""
    return bool(_HUMAN_REQUEST_PATTERN.search(text))


def is_social_conversation(text: str) -> bool:
    """Identify harmless short interaction that should not become a human handoff."""
    normalised = _normalise(text)
    if not normalised:
        return False
    if normalised in _GREETING_PHRASES or normalised in _HELP_PHRASES:
        return True
    words = normalised.split()
    if words and all(word in _GREETING_WORDS for word in words):
        return True
    return normalised.startswith("my name is ") or normalised.startswith("i am ") or normalised.startswith("im ")


def build_conversation_response(state: ConversationState) -> dict[str, object]:
    """Write a fixed non-factual greeting/capability response for the grounding gate."""
    text = _normalise(state["current_transcript"])
    if text.startswith("my name is ") or text.startswith("i am ") or text.startswith("im "):
        answer = _INTRODUCTION_RESPONSE
    elif text in _HELP_PHRASES:
        answer = _CAPABILITY_RESPONSE
    else:
        answer = _GREETING_RESPONSE
    call_log(state["session_id"], "CONVERSATION", "complete", details={"response_length": len(answer)})
    return {
        "draft_answer": answer,
        "retrieved_excerpts": [],
        "response_mode": "conversation",
        "system_failure": None,
    }


def build_clarification_response(state: ConversationState) -> dict[str, object]:
    """Ask a bounded clarification question before escalating a vague support request."""
    failed_clarification = state["awaiting_clarification"]
    clarification_count = state["clarification_count"] + int(failed_clarification)
    answer = _FOLLOW_UP_CLARIFICATION_RESPONSE if failed_clarification else _CLARIFICATION_RESPONSE
    call_log(
        state["session_id"],
        "CLARIFICATION",
        "complete",
        details={
            "clarification_count": clarification_count,
            "failed_clarification": failed_clarification,
        },
    )
    return {
        "draft_answer": answer,
        "retrieved_excerpts": [],
        "clarification_count": clarification_count,
        "awaiting_clarification": True,
        "clarification_topic": "support_category",
        "response_mode": "clarification",
        "system_failure": None,
    }


def is_deterministic_conversation_response(state: ConversationState) -> bool:
    """Restrict direct grounding approval to this module's fixed, non-factual templates."""
    return state["response_mode"] in {"conversation", "clarification"} and state["draft_answer"] in {
        _GREETING_RESPONSE,
        _INTRODUCTION_RESPONSE,
        _CAPABILITY_RESPONSE,
        _CLARIFICATION_RESPONSE,
        _FOLLOW_UP_CLARIFICATION_RESPONSE,
    }
