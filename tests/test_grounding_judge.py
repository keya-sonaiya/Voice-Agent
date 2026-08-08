from app.graph.nodes import grounding_judge

from .helpers import state


class FakeClient:
    def __init__(self, **_: object) -> None:
        pass

    def chat(self, **_: object) -> dict[str, dict[str, str]]:
        return {"message": {"content": '{"is_grounded":true,"reason":"Supported","cited_sources":["shipping.md"]}'}}


def test_grounding_result_is_schema_validated(monkeypatch: object) -> None:
    monkeypatch.setattr(grounding_judge, "Client", FakeClient)
    result = grounding_judge.check_grounding(
        {
            **state(),
            "draft_answer": "Orders process in one business day.",
            "retrieved_excerpts": ["Orders process in one business day."],
        }
    )
    assert result["grounding_result"].is_grounded


def test_malformed_judge_response_never_passes(monkeypatch: object) -> None:
    class BadClient(FakeClient):
        def chat(self, **_: object) -> dict[str, dict[str, str]]:
            return {"message": {"content": "{}"}}

    monkeypatch.setattr(grounding_judge, "Client", BadClient)
    result = grounding_judge.check_grounding(
        {**state(), "draft_answer": "An unsupported fact.", "retrieved_excerpts": ["A fact"]}
    )
    assert not result["grounding_result"].is_grounded
