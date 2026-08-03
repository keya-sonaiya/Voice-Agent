"""The validated state contract shared by every graph node."""

from datetime import datetime
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: Literal["caller", "agent"]
    text: str
    timestamp: datetime
    sentiment_score: Optional[float] = None


class IntentResult(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class GroundingResult(BaseModel):
    is_grounded: bool
    reason: str
    cited_sources: list[str] = Field(default_factory=list)


class EscalationDecision(BaseModel):
    should_escalate: bool
    reason: Literal[
        "low_confidence",
        "negative_sentiment_trend",
        "repeated_clarification",
        "explicit_human_request",
        "grounding_failure",
        "none",
    ]


class ConversationState(TypedDict):
    session_id: str
    turns: list[Turn]
    current_transcript: str
    intent_result: Optional[IntentResult]
    rolling_sentiment: float
    clarification_count: int
    draft_answer: Optional[str]
    grounding_result: Optional[GroundingResult]
    escalation_decision: Optional[EscalationDecision]
    final_response_text: Optional[str]
