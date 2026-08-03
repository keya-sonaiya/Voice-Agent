"""Schema-constrained intent classification node."""

from typing import Any, cast

from ollama import Client
from pydantic import ValidationError

from app.config import settings
from app.graph.state import ConversationState, IntentResult

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
    },
    "required": ["intent", "confidence"],
}
SYSTEM_PROMPT = """You classify customer support call intent. Return the intent label,
a confidence score 0-1, and one-sentence reasoning. Do not invent intents outside
billing, technical_issue, account_access, order_status, cancellation, complaint,
general_inquiry, human_request. Ambiguous requests must have low confidence."""


def classify_intent(state: ConversationState) -> dict[str, IntentResult]:
    """Read `current_transcript`; write `intent_result`; never route or answer callers."""
    client = Client(
        host=settings.ollama_host,
        headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
    )
    try:
        response = client.chat(
            model=settings.intent_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state["current_transcript"]},
            ],
            format=cast(Any, INTENT_SCHEMA),
        )
        result = IntentResult.model_validate_json(response["message"]["content"])
    except (KeyError, ValidationError, ValueError, OSError) as error:
        result = IntentResult(intent="general_inquiry", confidence=0.0, reasoning=str(error))
    return {"intent_result": result}
