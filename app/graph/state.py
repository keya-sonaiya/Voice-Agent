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
        "system_failure",
        "none",
    ]


class ConversationState(TypedDict):
    session_id: str
    authenticated_caller_id: Optional[str]
    turns: list[Turn]
    current_transcript: str
    intent_result: Optional[IntentResult]
    previous_intent: Optional[IntentResult]
    rolling_sentiment: float
    clarification_count: int
    awaiting_clarification: bool
    clarification_topic: Optional[str]
    clarification_resolved: bool
    customer_id: Optional[str]
    customer_identified: bool
    customer_verified: bool
    identity_state: Literal["unidentified", "identified", "verified"]
    verification_method: Optional[str]
    verification_timestamp: Optional[str]
    awaiting_customer_name: bool
    awaiting_customer_phone: bool
    awaiting_customer_email: bool
    account_recovery_active: bool
    account_recovery_attempts: int
    recovery_candidate_ids: list[str]
    support_intent: Optional[str]
    current_payment_id: Optional[str]
    current_invoice_id: Optional[str]
    current_order_id: Optional[str]
    current_ticket_id: Optional[str]
    awaiting_customer_verification: bool
    awaiting_payment_id: bool
    draft_answer: Optional[str]
    retrieved_excerpts: list[str]
    grounding_result: Optional[GroundingResult]
    escalation_decision: Optional[EscalationDecision]
    final_response_text: Optional[str]
    system_failure: Optional[str]
    response_mode: Optional[Literal["knowledge", "conversation", "clarification", "support_workflow"]]
