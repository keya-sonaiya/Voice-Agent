"""Schema-constrained hard safety gate for RAG answers."""

from typing import Any, cast

from ollama import Client
from pydantic import ValidationError

from app.config import settings
from app.graph.state import ConversationState, GroundingResult

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_grounded": {"type": "boolean"},
        "reason": {"type": "string"},
        "cited_sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["is_grounded", "reason"],
}
JUDGE_PROMPT = """You are a strict fact-checker. Given a customer question, retrieved
knowledge-base excerpts, and a draft answer, determine whether every factual claim
in the answer is directly supported by the excerpts. Any unsupported claim means false."""


def check_grounding(state: ConversationState) -> dict[str, GroundingResult]:
    """Read the transcript and draft; write `grounding_result`; never approve on validation error."""
    excerpts = state["retrieved_excerpts"]
    excerpts_text = "\n---\n".join(excerpts)
    payload = (
        f"Question: {state['current_transcript']}\n\nRetrieved excerpts:\n"
        f"{excerpts_text}\n\nDraft answer: {state['draft_answer']}"
    )
    client = Client(
        host=settings.ollama_host,
        headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
    )
    try:
        response = client.chat(
            model=settings.grounding_judge_model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": payload},
            ],
            format=cast(Any, JUDGE_SCHEMA),
        )
        result = GroundingResult.model_validate_json(response["message"]["content"])
    except (KeyError, ValidationError, ValueError, OSError) as error:
        result = GroundingResult(is_grounded=False, reason=f"Grounding verification unavailable: {error}")
    return {"grounding_result": result}
