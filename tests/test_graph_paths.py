from app.graph import build_graph as graph_module
from app.graph.nodes import sentiment_tracker
from app.graph.state import GroundingResult, IntentResult

from .helpers import state


def _graph(
    monkeypatch: object, intent: IntentResult, grounding: GroundingResult | None = None
) -> tuple[object, list[str]]:
    transitions: list[str] = []
    monkeypatch.setattr(graph_module, "record_transition", lambda name, _: transitions.append(name))
    monkeypatch.setattr(graph_module.intent_agent, "classify_intent", lambda _: {"intent_result": intent})
    monkeypatch.setattr(
        graph_module.knowledge_agent, "generate_answer", lambda _: {"draft_answer": "Supported answer."}
    )
    if grounding is not None:
        monkeypatch.setattr(graph_module.grounding_judge, "check_grounding", lambda _: {"grounding_result": grounding})
    return graph_module.build_graph(), transitions


def test_grounded_high_confidence_turn_reaches_response(monkeypatch: object) -> None:
    graph, transitions = _graph(
        monkeypatch,
        IntentResult(intent="order_status", confidence=0.95),
        GroundingResult(is_grounded=True, reason="Supported"),
    )
    result = graph.invoke(state())
    assert result["final_response_text"] == "Supported answer."
    assert transitions == ["intent", "sentiment", "knowledge", "grounding", "respond"]


def test_ambiguous_turn_escalates_for_low_confidence(monkeypatch: object) -> None:
    graph, _ = _graph(monkeypatch, IntentResult(intent="general_inquiry", confidence=0.2))
    result = graph.invoke(state("Can you help me with that thing?"))
    assert result["escalation_decision"].reason == "low_confidence"


def test_negative_sentiment_escalates_before_knowledge(monkeypatch: object) -> None:
    graph, transitions = _graph(monkeypatch, IntentResult(intent="complaint", confidence=0.99))
    result = graph.invoke(state("This service is awful and terrible."))
    assert result["rolling_sentiment"] < 0
    assert result["escalation_decision"].reason == "negative_sentiment_trend"
    assert transitions == ["intent", "sentiment", "escalation"]


def test_unsupported_draft_escalates_after_grounding_gate(monkeypatch: object) -> None:
    graph, _ = _graph(
        monkeypatch,
        IntentResult(intent="billing", confidence=0.99),
        GroundingResult(is_grounded=False, reason="Unsupported claim"),
    )
    result = graph.invoke(state())
    assert result["final_response_text"] is None
    assert result["escalation_decision"].reason == "grounding_failure"


def test_repeated_clarifications_reach_deterministic_escalation(monkeypatch: object) -> None:
    graph, _ = _graph(monkeypatch, IntentResult(intent="general_inquiry", confidence=0.99))
    result = graph.invoke({**state(), "clarification_count": 3})
    assert result["escalation_decision"].reason == "repeated_clarification"


def test_sentiment_node_remains_deterministic() -> None:
    result = sentiment_tracker.update_sentiment(state("Thanks, that was helpful."))
    assert result["rolling_sentiment"] > 0
