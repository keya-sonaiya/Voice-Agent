from app.graph.nodes import knowledge_agent, sentiment_tracker

from .helpers import state


def test_sentiment_tracks_negative_callers() -> None:
    result = sentiment_tracker.update_sentiment(state("This is awful and terrible"))
    assert result["rolling_sentiment"] < 0
    assert result["turns"][-1].sentiment_score is not None


def test_knowledge_error_does_not_expose_provider_error(monkeypatch: object) -> None:
    class FailingClient:
        def __init__(self, **_: object) -> None:
            pass

        def chat(self, **_: object) -> object:
            raise OSError("secret backend error")

    monkeypatch.setattr(knowledge_agent, "Client", FailingClient)
    monkeypatch.setattr(knowledge_agent, "retrieve", lambda _: ["Orders process in one business day."])
    assert "secret" not in knowledge_agent.generate_answer(state())["draft_answer"]


def test_knowledge_uses_retrieved_context(monkeypatch: object) -> None:
    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def chat(self, **_: object) -> dict[str, dict[str, str]]:
            return {"message": {"content": "Orders process in one business day."}}

    monkeypatch.setattr(knowledge_agent, "Client", FakeClient)
    monkeypatch.setattr(knowledge_agent, "retrieve", lambda _: ["Orders process in one business day."])
    assert knowledge_agent.generate_answer(state())["draft_answer"] == "Orders process in one business day."
