from datetime import UTC, datetime

from app.graph import build_graph as graph_module
from app.graph.nodes import intent_agent, sentiment_tracker
from app.graph.state import GroundingResult, IntentResult, Turn

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


def test_payment_problem_beginning_with_i_am_routes_to_knowledge_not_an_introduction(monkeypatch: object) -> None:
    graph, transitions = _graph(
        monkeypatch,
        IntentResult(intent="billing", confidence=0.98),
        GroundingResult(is_grounded=True, reason="Supported"),
    )
    result = graph.invoke(state("I am having some trouble with my payments."))
    assert result["final_response_text"] == "Supported answer."
    assert result["response_mode"] != "conversation"
    assert transitions == ["intent", "sentiment", "knowledge", "grounding", "respond"]


def test_hello_reaches_conversation_response_without_escalation(monkeypatch: object) -> None:
    graph, transitions = _graph(monkeypatch, IntentResult(intent="general_inquiry", confidence=0.3))
    result = graph.invoke(state("Hello hello hello hello"))
    assert result["final_response_text"] == "Hello! How can I help you today?"
    assert result["escalation_decision"] is None
    assert transitions == ["intent", "sentiment", "conversation", "grounding", "respond"]


def test_intro_with_uncertain_stt_text_reaches_conversation_response(monkeypatch: object) -> None:
    graph, _ = _graph(monkeypatch, IntentResult(intent="general_inquiry", confidence=0.2))
    result = graph.invoke(state("My name is I'm a song"))
    assert result["final_response_text"] == "Nice to meet you. How can I help you today?"
    assert result["escalation_decision"] is None


def test_ambiguous_low_confidence_turn_asks_clarification_first(monkeypatch: object) -> None:
    graph, _ = _graph(monkeypatch, IntentResult(intent="general_inquiry", confidence=0.2))
    result = graph.invoke(state("My problem is with something I bought."))
    assert result["final_response_text"].startswith("Sure. Could you tell me whether")
    assert result["clarification_count"] == 0
    assert result["awaiting_clarification"] is True
    assert result["escalation_decision"] is None


def test_resolved_payment_clarification_reaches_knowledge_without_repeating_or_escalating(monkeypatch: object) -> None:
    transitions: list[str] = []
    monkeypatch.setattr(graph_module, "record_transition", lambda name, _: transitions.append(name))
    monkeypatch.setattr(
        graph_module.knowledge_agent,
        "generate_answer",
        lambda _: {"draft_answer": "Your payment issue can be reviewed."},
    )
    monkeypatch.setattr(
        graph_module.grounding_judge,
        "check_grounding",
        lambda _: {"grounding_result": GroundingResult(is_grounded=True, reason="Supported")},
    )
    result = graph_module.build_graph().invoke(
        {**state("payments"), "awaiting_clarification": True, "clarification_count": 1}
    )
    assert result["intent_result"].intent == "billing"
    assert result["clarification_count"] == 0
    assert result["awaiting_clarification"] is False
    assert result["escalation_decision"] is None
    assert result["final_response_text"] == "Your payment issue can be reviewed."
    assert "clarification" not in transitions
    assert transitions == ["intent", "sentiment", "knowledge", "grounding", "respond"]


def test_order_tracking_reaches_knowledge_without_escalation(monkeypatch: object) -> None:
    transitions: list[str] = []
    monkeypatch.setattr(graph_module, "record_transition", lambda name, _: transitions.append(name))
    monkeypatch.setattr(
        graph_module.knowledge_agent, "generate_answer", lambda _: {"draft_answer": "Your order is in transit."}
    )
    monkeypatch.setattr(
        graph_module.grounding_judge,
        "check_grounding",
        lambda _: {"grounding_result": GroundingResult(is_grounded=True, reason="Supported")},
    )
    result = graph_module.build_graph().invoke(state("order tracking"))
    assert result["intent_result"].intent == "order_status"
    assert result["escalation_decision"] is None
    assert result["final_response_text"] == "Your order is in transit."
    assert transitions == ["intent", "sentiment", "knowledge", "grounding", "respond"]


def test_vague_then_payments_then_order_tracking_keeps_conversational_context(monkeypatch: object) -> None:
    class LowConfidenceClient:
        def __init__(self, **_: object) -> None:
            pass

        def chat(self, **_: object) -> dict[str, dict[str, str]]:
            return {
                "message": {
                    "content": (
                        '{"intent":"general_inquiry","confidence":0.2,'
                        '"reasoning":"The request lacks a support category."}'
                    )
                }
            }

    monkeypatch.setattr(intent_agent, "Client", LowConfidenceClient)
    monkeypatch.setattr(
        graph_module.knowledge_agent,
        "generate_answer",
        lambda _: {"draft_answer": "A supported help answer.", "response_mode": "knowledge"},
    )
    monkeypatch.setattr(
        graph_module.grounding_judge,
        "check_grounding",
        lambda _: {"grounding_result": GroundingResult(is_grounded=True, reason="Supported")},
    )
    graph = graph_module.build_graph()

    first = graph.invoke(state("Hello, I am some troubles."))
    assert first["response_mode"] == "clarification"
    assert first["awaiting_clarification"] is True
    assert first["clarification_count"] == 0

    agent_turn = Turn(role="agent", text=first["final_response_text"], timestamp=datetime.now(UTC))
    second = graph.invoke(
        {
            **first,
            "current_transcript": "payments",
            "previous_intent": first["intent_result"],
            "clarification_resolved": False,
            "turns": [*first["turns"], agent_turn, Turn(role="caller", text="payments", timestamp=datetime.now(UTC))],
        }
    )
    assert second["intent_result"].intent == "billing"
    assert second["response_mode"] == "knowledge"
    assert second["awaiting_clarification"] is False
    assert second["clarification_count"] == 0
    assert second["escalation_decision"] is None
    assert second["final_response_text"] != first["final_response_text"]

    billing_turn = Turn(role="agent", text=second["final_response_text"], timestamp=datetime.now(UTC))
    third = graph.invoke(
        {
            **second,
            "current_transcript": "order tracking",
            "previous_intent": second["intent_result"],
            "clarification_resolved": False,
            "turns": [
                *second["turns"],
                billing_turn,
                Turn(role="caller", text="order tracking", timestamp=datetime.now(UTC)),
            ],
        }
    )
    assert third["intent_result"].intent == "order_status"
    assert third["escalation_decision"] is None
    assert third["final_response_text"] == "A supported help answer."


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
    graph, _ = _graph(monkeypatch, IntentResult(intent="general_inquiry", confidence=0.2))
    result = graph.invoke({**state("A vague issue"), "clarification_count": 2, "awaiting_clarification": True})
    assert result["escalation_decision"].reason == "repeated_clarification"


def test_explicit_human_request_escalates_even_if_intent_is_uncertain(monkeypatch: object) -> None:
    graph, transitions = _graph(monkeypatch, IntentResult(intent="general_inquiry", confidence=0.1))
    result = graph.invoke(state("I want a human"))
    assert result["escalation_decision"].reason == "explicit_human_request"
    assert transitions == ["intent", "escalation"]


def test_sentiment_node_remains_deterministic() -> None:
    result = sentiment_tracker.update_sentiment(state("Thanks, that was helpful."))
    assert result["rolling_sentiment"] > 0
