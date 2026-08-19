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


def test_boolean_judge_response_is_a_visible_system_failure(monkeypatch: object, caplog: object) -> None:
    class BooleanClient(FakeClient):
        def chat(self, **_: object) -> dict[str, dict[str, str]]:
            return {"message": {"content": "True"}}

    monkeypatch.setattr(grounding_judge, "Client", BooleanClient)
    result = grounding_judge.check_grounding(
        {**state(), "draft_answer": "Orders process in one business day.", "retrieved_excerpts": ["A fact"]}
    )
    assert result["system_failure"] == "grounding"
    assert not result["grounding_result"].is_grounded
    assert result["grounding_result"].reason == "Grounding judge returned invalid structured output."
    assert "[event=structured_output_invalid]" in caplog.text


def test_sole_json_markdown_fence_is_unwrapped_then_schema_validated(monkeypatch: object) -> None:
    class FencedJsonClient(FakeClient):
        def chat(self, **_: object) -> dict[str, dict[str, str]]:
            return {
                "message": {
                    "content": (
                        "```json\n" '{"is_grounded":true,"reason":"Supported","cited_sources":["shipping.md"]}' "\n```"
                    )
                }
            }

    monkeypatch.setattr(grounding_judge, "Client", FencedJsonClient)
    result = grounding_judge.check_grounding(
        {**state(), "draft_answer": "Orders process in one business day.", "retrieved_excerpts": ["A fact"]}
    )
    assert result["system_failure"] is None
    assert result["grounding_result"].is_grounded
