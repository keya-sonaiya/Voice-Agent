from app.graph.nodes.escalation_agent import decide_escalation
from app.graph.state import GroundingResult, IntentResult

from .helpers import state


def test_low_confidence_escalates() -> None:
    result = decide_escalation({**state(), "intent_result": IntentResult(intent="billing", confidence=0.2)})
    assert result["escalation_decision"].reason == "low_confidence"


def test_negative_sentiment_escalates() -> None:
    result = decide_escalation(
        {
            **state(),
            "intent_result": IntentResult(intent="billing", confidence=0.99),
            "rolling_sentiment": -0.5,
        }
    )
    assert result["escalation_decision"].reason == "negative_sentiment_trend"


def test_grounding_failure_has_priority() -> None:
    result = decide_escalation(
        {**state(), "grounding_result": GroundingResult(is_grounded=False, reason="Unsupported")}
    )
    assert result["escalation_decision"].reason == "grounding_failure"


def test_human_request_escalates() -> None:
    result = decide_escalation({**state(), "intent_result": IntentResult(intent="human_request", confidence=1)})
    assert result["escalation_decision"].reason == "explicit_human_request"
