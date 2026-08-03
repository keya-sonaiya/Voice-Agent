from app.graph.nodes import intent_agent

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
    result = intent_agent.classify_intent(state())
    assert result["intent_result"].intent == "order_status"
    assert result["intent_result"].confidence == 0.91


def test_malformed_intent_response_fails_closed(monkeypatch: object) -> None:
    class BadClient(FakeClient):
        def chat(self, **_: object) -> dict[str, dict[str, str]]:
            return {"message": {"content": "not-json"}}

    monkeypatch.setattr(intent_agent, "Client", BadClient)
    assert intent_agent.classify_intent(state())["intent_result"].confidence == 0.0
