"""Schema-constrained intent classification node."""

import re
from time import monotonic
from typing import Any, cast

from ollama import Client
from pydantic import ValidationError

from app.call_logging import call_exception, call_log, duration_ms
from app.config import settings
from app.graph.state import ConversationState, IntentResult

_LLM_CLIENT: Client | None = None
_LLM_CLIENT_CLASS: type[Client] | None = None


def get_llm_client() -> Client:
    global _LLM_CLIENT, _LLM_CLIENT_CLASS
    if _LLM_CLIENT is None or _LLM_CLIENT_CLASS is not Client:
        _LLM_CLIENT = Client(host=settings.ollama_host, headers={"Authorization": f"Bearer {settings.ollama_api_key}"})
        _LLM_CLIENT_CLASS = Client
    return _LLM_CLIENT

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
general_inquiry, human_request. Use the conversation context: short caller replies can
answer the agent's immediately preceding clarification question. Ambiguous requests
must have low confidence."""

_INTENT_LABELS = {
    "billing",
    "technical_issue",
    "account_access",
    "order_status",
    "cancellation",
    "complaint",
    "general_inquiry",
    "human_request",
}
_SUPPORT_PHRASE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "order_status",
        re.compile(
            r"\b(?:order\s*(?:tracking|status)|track(?:ing)?\s+(?:my\s+)?(?:order|package)|"
            r"where(?:'s|\s+is)\s+my\s+(?:order|package)|package\s+tracking)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "billing",
        re.compile(
            r"\b(?:payment|payments|billing|bill|charged|charge|wrong\s+charge|invoice)\b",
            re.IGNORECASE,
        ),
    ),
    ("cancellation", re.compile(r"\b(?:cancel|cancellation|refund)\b", re.IGNORECASE)),
    (
        "account_access",
        re.compile(r"\b(?:account|login|log\s*in|password|sign\s*in|access)\b", re.IGNORECASE),
    ),
    (
        "technical_issue",
        re.compile(r"\b(?:technical|error|broken|not\s+working|bug|app\s+issue)\b", re.IGNORECASE),
    ),
)


def _validate_intent_label(result: IntentResult) -> IntentResult:
    if result.intent not in _INTENT_LABELS:
        raise ValueError("Intent response used an unsupported label.")
    return result


def _parse_intent(content: object) -> IntentResult:
    """Validate JSON, with a narrow fallback for providers that ignore ``format``."""
    if not isinstance(content, str):
        raise ValueError("Intent response content is missing.")
    try:
        return _validate_intent_label(IntentResult.model_validate_json(content))
    except ValidationError as json_error:
        intent_match = re.search(r"(?im)^\s*intent\s*:\s*['\"]?([a-z_]+)", content)
        confidence_match = re.search(r"(?im)^\s*confidence\s*:\s*([01](?:\.\d+)?)", content)
        reasoning_match = re.search(r"(?im)^\s*reasoning\s*:\s*(.+)$", content)
        if intent_match is None or confidence_match is None:
            raise json_error
        intent = intent_match.group(1)
        return _validate_intent_label(
            IntentResult.model_validate(
                {
                    "intent": intent,
                    "confidence": float(confidence_match.group(1)),
                    "reasoning": reasoning_match.group(1).strip(" '\"") if reasoning_match else None,
                }
            )
        )


def _compact_conversation_context(state: ConversationState) -> str:
    """Keep the intent prompt small while retaining the turns that explain short replies."""
    recent_turns = state["turns"][-6:]
    lines = ["CONVERSATION:"]
    for turn in recent_turns:
        role = "Caller" if turn.role == "caller" else "Agent"
        lines.append(f"{role}: {turn.text[:500]}")
    previous_intent = state["previous_intent"] or state["intent_result"]
    if previous_intent is not None:
        lines.append(f"Previous intent: {previous_intent.intent} (confidence {previous_intent.confidence:.2f})")
    lines.append(f"Awaiting clarification: {state['awaiting_clarification']}")
    lines.append(f"Clarification count: {state['clarification_count']}")
    lines.append(f"Current caller transcript: {state['current_transcript'][:500]}")
    return "\n".join(lines)


def _classify_known_support_phrase(transcript: str) -> IntentResult | None:
    """Resolve unmistakable taxonomy phrases without requiring provider interpretation."""
    if re.search(r"\b(?:forgot|don't know|do not know|lost)\b.{0,30}\b(?:customer|account)\s+id\b", transcript, re.I):
        return IntentResult(
            intent="account_access",
            confidence=0.99,
            reasoning="Matched deterministic customer-ID recovery phrase.",
        )
    for intent, pattern in _SUPPORT_PHRASE_PATTERNS:
        if pattern.search(transcript):
            return IntentResult(
                intent=intent,
                confidence=0.98,
                reasoning="Matched an established customer-support intent phrase.",
            )
    return None


def classify_intent(state: ConversationState) -> dict[str, object]:
    """Classify the caller intent and distinguish provider failures from ambiguity."""
    started = monotonic()
    call_log(
        state["session_id"],
        "INTENT",
        "start",
        details={"transcript_length": len(state["current_transcript"]), "model": settings.intent_model},
    )
    previous_intent = state["previous_intent"] or state["intent_result"]
    call_log(
        state["session_id"],
        "INTENT",
        "context_used",
        details={
            "previous_intent": previous_intent.intent if previous_intent else "none",
            "awaiting_clarification": state["awaiting_clarification"],
            "clarification_count": state["clarification_count"],
        },
    )
    if state["support_intent"] in _INTENT_LABELS:
        result = IntentResult(
            intent=state["support_intent"],
            confidence=0.99,
            reasoning="Continuing a server-controlled customer-support workflow.",
        )
        return {
            "intent_result": result,
            "previous_intent": previous_intent,
            "clarification_resolved": False,
            "system_failure": None,
        }
    deterministic_result = _classify_known_support_phrase(state["current_transcript"])
    if deterministic_result is not None:
        clarification_resolved = state["awaiting_clarification"]
        if clarification_resolved:
            call_log(
                state["session_id"],
                "INTENT",
                "clarification_resolved",
                details={
                    "user_input": state["current_transcript"],
                    "resolved_intent": deterministic_result.intent,
                    "confidence": deterministic_result.confidence,
                },
            )
        return {
            "intent_result": deterministic_result,
            "previous_intent": previous_intent,
            "awaiting_clarification": False,
            "clarification_count": 0,
            "clarification_topic": deterministic_result.intent,
            "clarification_resolved": clarification_resolved,
            "system_failure": None,
        }
    try:
        response = get_llm_client().chat(
            model=settings.intent_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _compact_conversation_context(state)},
            ],
            format=cast(Any, INTENT_SCHEMA),
        )
        result = _parse_intent(response["message"]["content"])
    except Exception:
        # Never turn an unavailable model into a seemingly ordinary low-confidence turn.
        call_exception(state["session_id"], "INTENT", "failed", details={"model": settings.intent_model})
        return {
            "intent_result": IntentResult(
                intent="general_inquiry", confidence=0.0, reasoning="Intent classification unavailable."
            ),
            "previous_intent": previous_intent,
            "clarification_resolved": False,
            "system_failure": "intent",
        }
    clarification_resolved = (
        state["awaiting_clarification"]
        and result.intent != "general_inquiry"
        and result.confidence >= settings.confidence_threshold
    )
    if clarification_resolved:
        call_log(
            state["session_id"],
            "INTENT",
            "clarification_resolved",
            details={
                "user_input": state["current_transcript"],
                "resolved_intent": result.intent,
                "confidence": result.confidence,
            },
        )
    call_log(
        state["session_id"],
        "INTENT",
        "complete",
        duration=duration_ms(started),
        details={"intent": result.intent, "confidence": result.confidence},
    )
    return {
        "intent_result": result,
        "previous_intent": previous_intent,
        "awaiting_clarification": False if clarification_resolved else state["awaiting_clarification"],
        "clarification_count": 0 if clarification_resolved else state["clarification_count"],
        "clarification_topic": result.intent if clarification_resolved else state["clarification_topic"],
        "clarification_resolved": clarification_resolved,
        "system_failure": None,
    }
