"""Schema-constrained hard safety gate for RAG answers."""

import logging
import re
from time import monotonic
from typing import Any, cast

from ollama import Client
from pydantic import ValidationError

from app.call_logging import call_exception, call_log, duration_ms
from app.config import settings
from app.graph.nodes.conversation_agent import is_deterministic_conversation_response
from app.graph.nodes.support_workflow import is_deterministic_support_prompt
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
in the answer is directly supported by the excerpts. Any unsupported claim means false.

Return only a JSON object that conforms exactly to the supplied JSON schema. Do not
return a boolean, Markdown, prose, code fences, or additional keys. Include both
is_grounded and reason."""

_SOLE_JSON_FENCE = re.compile(r"\A\s*```json\s*\n(?P<document>\{.*\})\s*```\s*\Z", re.DOTALL | re.IGNORECASE)


def _output_metadata(content: object) -> dict[str, object]:
    """Return a redacted, bounded description of an untrusted model response for logs."""
    if isinstance(content, str):
        return {
            "content_type": type(content).__name__,
            "content_length": len(content),
            "preview": content[:200],
        }
    return {
        "content_type": type(content).__name__,
        "content_length": None,
        "preview": str(content)[:200],
    }


def _unwrap_sole_json_fence(content: object) -> tuple[object, bool]:
    """Remove only a complete JSON Markdown fence; never interpret arbitrary model prose."""
    if not isinstance(content, str):
        return content, False
    match = _SOLE_JSON_FENCE.match(content)
    if match is None:
        return content, False
    return match.group("document"), True


def check_grounding(state: ConversationState) -> dict[str, object]:
    """Approve only a schema-valid grounded answer; surface judge outages separately."""
    started = monotonic()
    excerpts = state["retrieved_excerpts"]
    call_log(
        state["session_id"],
        "GROUNDING",
        "start",
        details={
            "excerpt_count": len(excerpts),
            "draft_answer_length": len(state["draft_answer"] or ""),
            "model": settings.grounding_judge_model,
        },
    )
    if is_deterministic_conversation_response(state) or is_deterministic_support_prompt(state):
        result = GroundingResult(
            is_grounded=True,
            reason="Approved deterministic non-factual conversational template.",
        )
        call_log(
            state["session_id"],
            "GROUNDING",
            "complete",
            duration=duration_ms(started),
            details={"is_grounded": True, "reason": result.reason, "mode": state["response_mode"]},
        )
        return {"grounding_result": result, "system_failure": None}
    excerpts_text = "\n---\n".join(excerpts)
    payload = (
        f"Question: {state['current_transcript']}\n\nRetrieved excerpts:\n"
        f"{excerpts_text}\n\nDraft answer: {state['draft_answer']}"
    )
    try:
        client = Client(
            host=settings.ollama_host,
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
        )
        call_log(
            state["session_id"],
            "GROUNDING",
            "model_request",
            details={"model": settings.grounding_judge_model, "structured_output": True, "schema_mode": "json_schema"},
        )
        response = client.chat(
            model=settings.grounding_judge_model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": payload},
            ],
            format=cast(Any, JUDGE_SCHEMA),
            options={"temperature": 0},
        )
        raw_content = response["message"]["content"]
        output_metadata = _output_metadata(raw_content)
        call_log(state["session_id"], "GROUNDING", "raw_model_output", details=output_metadata)
        content, unwrapped_fence = _unwrap_sole_json_fence(raw_content)
        if unwrapped_fence:
            call_log(
                state["session_id"],
                "GROUNDING",
                "structured_output_fence_removed",
                details={"model": settings.grounding_judge_model, "schema_mode": "json_schema"},
            )
        try:
            if not isinstance(content, (str, bytes, bytearray)):
                raise TypeError("Grounding model response content must be JSON text.")
            result = GroundingResult.model_validate_json(content)
        except (TypeError, ValidationError) as error:
            call_log(
                state["session_id"],
                "GROUNDING",
                "structured_output_invalid",
                level=logging.ERROR,
                details={
                    "model": settings.grounding_judge_model,
                    "validation_error": str(error),
                    **output_metadata,
                },
            )
            return {
                "grounding_result": GroundingResult(
                    is_grounded=False,
                    reason="Grounding judge returned invalid structured output.",
                ),
                "system_failure": "grounding",
            }
    except Exception:
        call_exception(state["session_id"], "GROUNDING", "failed", details={"model": settings.grounding_judge_model})
        return {
            "grounding_result": GroundingResult(is_grounded=False, reason="Grounding verification unavailable."),
            "system_failure": "grounding",
        }
    call_log(
        state["session_id"],
        "GROUNDING",
        "complete",
        duration=duration_ms(started),
        details={"is_grounded": result.is_grounded, "reason": result.reason},
    )
    return {"grounding_result": result, "system_failure": None}
