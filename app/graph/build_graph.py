"""LangGraph assembly with audited persistence after every transition."""

import logging
from collections.abc import Callable
from time import monotonic
from typing import Any, cast

from langgraph.graph import END, StateGraph

from app.call_logging import call_exception, call_log, duration_ms
from app.config import settings
from app.graph.nodes import (
    conversation_agent,
    escalation_agent,
    grounding_judge,
    intent_agent,
    knowledge_agent,
    sentiment_tracker,
    support_workflow,
)
from app.graph.state import ConversationState
from app.persistence.session_store import record_transition

Node = Callable[[ConversationState], dict[str, Any]]
logger = logging.getLogger(__name__)

_STAGES = {
    "intent": "INTENT",
    "sentiment": "SENTIMENT",
    "conversation": "CONVERSATION",
    "clarification": "CLARIFICATION",
    "knowledge": "RAG",
    "grounding": "GROUNDING",
    "escalation": "ESCALATION",
    "failure_terminal": "GRAPH",
    "respond": "RESPONSE",
}


def _persisted(name: str, node: Node) -> Node:
    """Add timing, persistence, and exception tracing without altering node ownership."""

    def wrapped(state: ConversationState) -> dict[str, Any]:
        stage = _STAGES[name]
        started = monotonic()
        call_log(state["session_id"], stage, "start")
        try:
            update = node(state)
        except Exception:
            call_exception(state["session_id"], stage, "failed")
            raise
        merged = dict(state)
        merged.update(update)
        try:
            record_transition(name, cast(ConversationState, merged))
        except Exception:
            # State persistence is audit evidence; it must not terminate the live conversation.
            call_exception(state["session_id"], "PERSISTENCE", "failed", details={"transition": name})
        call_log(state["session_id"], stage, "complete", duration=duration_ms(started))
        return update

    return wrapped


def route_after_intent(state: ConversationState) -> str:
    """Keep model outages and direct human requests terminal; let normal callers reach sentiment."""
    if state["system_failure"]:
        next_node = "failure_terminal"
        reason = "system_failure"
    elif conversation_agent.is_explicit_human_request(state["current_transcript"]):
        next_node = "escalation"
        reason = "explicit_human_request"
    else:
        next_node = "sentiment"
        reason = "clarification_resolved" if state["clarification_resolved"] else "continue_normal_turn"
    intent = state["intent_result"]
    call_log(
        state["session_id"],
        "ROUTER",
        "after_intent",
        details={
            "intent": intent.intent if intent else "missing",
            "confidence": intent.confidence if intent else "missing",
            "next": next_node,
            "reason": reason,
        },
    )
    return next_node


def route_after_sentiment(state: ConversationState) -> str:
    """Choose support/RAG, safe conversation, clarification, or deterministic escalation."""
    if state["system_failure"]:
        next_node = "failure_terminal"
        reason = "system_failure"
    elif conversation_agent.is_explicit_human_request(state["current_transcript"]):
        next_node = "escalation"
        reason = "explicit_human_request"
    elif state["rolling_sentiment"] < settings.sentiment_escalation_threshold:
        next_node = "escalation"
        reason = "negative_sentiment_trend"
    else:
        intent = state["intent_result"]
        low_confidence = intent is None or intent.confidence < settings.confidence_threshold
        if support_workflow.is_support_workflow_turn(state):
            next_node = "knowledge"
            reason = "customer_tool_workflow"
        elif not low_confidence:
            # A concrete support request takes precedence over a casual phrase such
            # as "I am having trouble with my payments".
            next_node = "knowledge"
            reason = "confident_support_request"
        elif conversation_agent.is_social_conversation(state["current_transcript"]):
            next_node = "conversation"
            reason = "safe_social_conversation"
        else:
            if state["awaiting_clarification"] and state["clarification_count"] >= settings.max_clarifications:
                next_node = "escalation"
                reason = "clarification_limit_reached"
            else:
                next_node = "clarification"
                reason = (
                    "low_confidence_after_clarification"
                    if state["awaiting_clarification"]
                    else "low_confidence_first_clarification"
                )
    intent = state["intent_result"]
    call_log(
        state["session_id"],
        "ROUTER",
        "after_sentiment",
        details={
            "intent": intent.intent if intent else "missing",
            "confidence": intent.confidence if intent else "missing",
            "rolling_sentiment": state["rolling_sentiment"],
            "clarification_count": state["clarification_count"],
            "awaiting_clarification": state["awaiting_clarification"],
            "next": next_node,
            "reason": reason,
        },
    )
    return next_node


def route_after_knowledge(state: ConversationState) -> str:
    """Never pass a missing or failed answer into the grounding model."""
    if state["system_failure"] or not isinstance(state["draft_answer"], str) or not state["draft_answer"].strip():
        next_node = "failure_terminal"
        reason = state["system_failure"] or "empty_draft_answer"
    else:
        next_node = "grounding"
        reason = "draft_answer_ready"
    call_log(state["session_id"], "ROUTER", "after_knowledge", details={"next": next_node, "reason": reason})
    return next_node


def route_after_grounding(state: ConversationState) -> str:
    """Enforce the hard grounding gate before the response terminal node."""
    if state["system_failure"]:
        next_node = "failure_terminal"
        reason = "system_failure"
    else:
        grounding = state["grounding_result"]
        if grounding is not None and grounding.is_grounded and state["draft_answer"]:
            next_node = "respond"
            reason = "grounding_passed"
        else:
            next_node = "escalation"
            reason = grounding.reason if grounding else "missing_grounding_result"
    call_log(state["session_id"], "ROUTER", "after_grounding", details={"next": next_node, "reason": reason})
    return next_node


def _respond(state: ConversationState) -> dict[str, str]:
    """Publish only the answer that has passed the grounding route."""
    answer = state["draft_answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Response node reached without a grounded draft answer.")
    call_log(state["session_id"], "GRAPH", "response_ready", details={"response_length": len(answer)})
    return {"final_response_text": answer}


def build_graph() -> Any:
    """Compile the mandatory multi-agent support pipeline with explicit safe terminals."""
    graph = StateGraph(ConversationState)
    graph.add_node("intent", _persisted("intent", intent_agent.classify_intent))
    graph.add_node("sentiment", _persisted("sentiment", sentiment_tracker.update_sentiment))
    graph.add_node("conversation", _persisted("conversation", conversation_agent.build_conversation_response))
    graph.add_node("clarification", _persisted("clarification", conversation_agent.build_clarification_response))
    graph.add_node("knowledge", _persisted("knowledge", knowledge_agent.generate_answer))
    graph.add_node("grounding", _persisted("grounding", grounding_judge.check_grounding))
    graph.add_node("escalation", _persisted("escalation", escalation_agent.decide_escalation))
    graph.add_node("failure_terminal", _persisted("failure_terminal", escalation_agent.decide_escalation))
    graph.add_node("respond", _persisted("respond", _respond))
    graph.set_entry_point("intent")
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {"sentiment": "sentiment", "escalation": "escalation", "failure_terminal": "failure_terminal"},
    )
    graph.add_conditional_edges(
        "sentiment",
        route_after_sentiment,
        {
            "knowledge": "knowledge",
            "conversation": "conversation",
            "clarification": "clarification",
            "escalation": "escalation",
            "failure_terminal": "failure_terminal",
        },
    )
    graph.add_edge("conversation", "grounding")
    graph.add_edge("clarification", "grounding")
    graph.add_conditional_edges(
        "knowledge",
        route_after_knowledge,
        {"grounding": "grounding", "failure_terminal": "failure_terminal"},
    )
    graph.add_conditional_edges(
        "grounding",
        route_after_grounding,
        {"respond": "respond", "escalation": "escalation", "failure_terminal": "failure_terminal"},
    )
    graph.add_edge("escalation", END)
    graph.add_edge("failure_terminal", END)
    graph.add_edge("respond", END)
    return graph.compile()
