"""LangGraph assembly with audited persistence after every transition."""

import logging

from collections.abc import Callable
from typing import Any, cast

from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    escalation_agent,
    grounding_judge,
    intent_agent,
    knowledge_agent,
    sentiment_tracker,
)
from app.graph.state import ConversationState
from app.persistence.session_store import record_transition

Node = Callable[[ConversationState], dict[str, Any]]
logger = logging.getLogger(__name__)


def _persisted(name: str, node: Node) -> Node:
    def wrapped(state: ConversationState) -> dict[str, Any]:
        update = node(state)
        merged = dict(state)
        merged.update(update)
        try:
            record_transition(name, cast(ConversationState, merged))
        except Exception:
            logger.exception("persistence_write_failed node=%s session_id=%s", name, state["session_id"])
        return update

    return wrapped


def route_after_intent(state: ConversationState) -> str:
    """Send any low-confidence or human-request intent to deterministic escalation."""
    decision = escalation_agent.decide_escalation(state)["escalation_decision"]
    return "escalation" if decision.should_escalate else "sentiment"


def route_after_sentiment(state: ConversationState) -> str:
    """Stop before RAG when the caller's rolling sentiment crosses the configured threshold."""
    decision = escalation_agent.decide_escalation(state)["escalation_decision"]
    return "escalation" if decision.should_escalate else "knowledge"


def route_after_grounding(state: ConversationState) -> str:
    """Enforce the hard grounding gate before the response terminal node."""
    grounding = state["grounding_result"]
    return "respond" if grounding is not None and grounding.is_grounded else "escalation"


def _respond(state: ConversationState) -> dict[str, str | None]:
    """Read grounded `draft_answer`; write `final_response_text`; never run before grounding."""
    return {"final_response_text": state["draft_answer"]}


def build_graph() -> Any:
    """Compile the multi-agent pipeline; all routes are explicit and auditable."""
    graph = StateGraph(ConversationState)
    graph.add_node("intent", _persisted("intent", intent_agent.classify_intent))
    graph.add_node("sentiment", _persisted("sentiment", sentiment_tracker.update_sentiment))
    graph.add_node("knowledge", _persisted("knowledge", knowledge_agent.generate_answer))
    graph.add_node("grounding", _persisted("grounding", grounding_judge.check_grounding))
    graph.add_node("escalation", _persisted("escalation", escalation_agent.decide_escalation))
    graph.add_node("respond", _persisted("respond", _respond))
    graph.set_entry_point("intent")
    graph.add_conditional_edges("intent", route_after_intent, {"sentiment": "sentiment", "escalation": "escalation"})
    graph.add_conditional_edges(
        "sentiment", route_after_sentiment, {"knowledge": "knowledge", "escalation": "escalation"}
    )
    graph.add_edge("knowledge", "grounding")
    graph.add_conditional_edges("grounding", route_after_grounding, {"respond": "respond", "escalation": "escalation"})
    graph.add_edge("escalation", END)
    graph.add_edge("respond", END)
    return graph.compile()
