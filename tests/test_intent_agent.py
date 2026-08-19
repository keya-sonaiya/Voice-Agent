from datetime import UTC, datetime

from app.graph.nodes import intent_agent
from app.graph.state import IntentResult, Turn

from .helpers import state


class FakeClient:
    def __init__(self, **_: object) -> None:
        pass

    def chat(self, **_: object) -> dict[str, dict[str, str]]:
        return {
            "message": {"content": '{"intent":"order_status","confidence":0.91,"reasoning":"Clear order question."}'}
        }


def test_intent_uses_constrained_schema_and_validates(monkeypatch: object) -> None:
    monkeypatch.setattr(intent_agent, "Client", FakeClient)
    result = intent_agent.classify_intent(state("Can you give me an update?"))
    assert result["intent_result"].intent == "order_status"
    assert result["intent_result"].confidence == 0.91


def test_malformed_intent_response_fails_closed(monkeypatch: object) -> None:
    class BadClient(FakeClient):
        def chat(self, **_: object) -> dict[str, dict[str, str]]:
            return {"message": {"content": "not-json"}}

    monkeypatch.setattr(intent_agent, "Client", BadClient)
    assert intent_agent.classify_intent(state("Can you give me an update?"))["intent_result"].confidence == 0.0


def test_payment_reply_resolves_an_outstanding_clarification_without_a_model_call(monkeypatch: object) -> None:
    class UnexpectedClient:
        def __init__(self, **_: object) -> None:
            raise AssertionError("Known clarification replies must not require a model call.")

    monkeypatch.setattr(intent_agent, "Client", UnexpectedClient)
    clarification = (
        "Sure. Could you tell me whether you need help with your order, payment, cancellation, account, "
        "or something else?"
    )
    result = intent_agent.classify_intent(
        {
            **state("payments"),
            "turns": [
                Turn(role="caller", text="I have some troubles.", timestamp=datetime.now(UTC)),
                Turn(role="agent", text=clarification, timestamp=datetime.now(UTC)),
                Turn(role="caller", text="payments", timestamp=datetime.now(UTC)),
            ],
            "awaiting_clarification": True,
            "clarification_count": 1,
            "previous_intent": IntentResult(intent="general_inquiry", confidence=0.2),
        }
    )
    assert result["intent_result"].intent == "billing"
    assert result["intent_result"].confidence >= 0.95
    assert result["awaiting_clarification"] is False
    assert result["clarification_count"] == 0
    assert result["clarification_resolved"] is True


def test_order_tracking_maps_to_existing_order_status_intent() -> None:
    result = intent_agent.classify_intent(state("order tracking"))
    assert result["intent_result"].intent == "order_status"
    assert result["intent_result"].confidence >= 0.95


def test_intent_model_receives_compact_recent_conversation_context(monkeypatch: object) -> None:
    captured: dict[str, object] = {}

    class ContextClient(FakeClient):
        def chat(self, **kwargs: object) -> dict[str, dict[str, str]]:
            captured.update(kwargs)
            return super().chat(**kwargs)

    monkeypatch.setattr(intent_agent, "Client", ContextClient)
    result = intent_agent.classify_intent(
        {
            **state("something else"),
            "turns": [
                Turn(role="caller", text="I have a problem.", timestamp=datetime.now(UTC)),
                Turn(role="agent", text="What kind of problem is it?", timestamp=datetime.now(UTC)),
                Turn(role="caller", text="something else", timestamp=datetime.now(UTC)),
            ],
            "awaiting_clarification": True,
            "clarification_count": 1,
            "previous_intent": IntentResult(intent="general_inquiry", confidence=0.2),
        }
    )
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "Caller: I have a problem." in messages[1]["content"]
    assert "Agent: What kind of problem is it?" in messages[1]["content"]
    assert "Awaiting clarification: True" in messages[1]["content"]
    assert result["intent_result"].intent == "order_status"
