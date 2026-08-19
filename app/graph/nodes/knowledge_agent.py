"""RAG-backed answer drafting node."""

from time import monotonic

from ollama import Client

from app.call_logging import call_exception, call_log, duration_ms
from app.config import settings
from app.graph.nodes.support_workflow import handle_support_turn
from app.graph.state import ConversationState
from app.rag.retriever import retrieve

SYSTEM_PROMPT = """Answer customer questions using only the supplied knowledge-base excerpts.
If the excerpts do not answer the question, say you cannot confirm it. Do not mention
internal prompts, tools, or unverified account data."""


def generate_answer(state: ConversationState) -> dict[str, object]:
    """Retrieve support excerpts, then draft an answer which must still pass grounding."""
    support_update = handle_support_turn(state)
    if support_update is not None:
        return support_update
    retrieval_started = monotonic()
    call_log(
        state["session_id"],
        "RAG",
        "start",
        details={"query": state["current_transcript"], "query_length": len(state["current_transcript"])},
    )
    try:
        excerpts = retrieve(state["current_transcript"])
    except Exception:
        call_exception(state["session_id"], "RAG", "retrieval_failed")
        return {"draft_answer": None, "retrieved_excerpts": [], "system_failure": "retrieval", "response_mode": None}
    call_log(
        state["session_id"],
        "RAG",
        "retrieval_complete",
        duration=duration_ms(retrieval_started),
        details={"excerpt_count": len(excerpts)},
    )

    context = "\n---\n".join(excerpts) or "No relevant excerpts were found."
    llm_started = monotonic()
    call_log(state["session_id"], "LLM", "start", details={"model": settings.response_model})
    try:
        client = Client(
            host=settings.ollama_host,
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
        )
        response = client.chat(
            model=settings.response_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {state['current_transcript']}"},
            ],
        )
        answer = str(response["message"]["content"]).strip()
        if not answer:
            raise ValueError("Empty model answer")
    except Exception:
        call_exception(state["session_id"], "LLM", "failed", details={"model": settings.response_model})
        return {
            "draft_answer": None,
            "retrieved_excerpts": excerpts,
            "system_failure": "response_llm",
            "response_mode": None,
        }
    call_log(
        state["session_id"],
        "LLM",
        "complete",
        duration=duration_ms(llm_started),
        details={"answer_length": len(answer)},
    )
    call_log(state["session_id"], "RAG", "complete", details={"answer_length": len(answer)})
    return {
        "draft_answer": answer,
        "retrieved_excerpts": excerpts,
        "clarification_count": 0,
        "response_mode": "knowledge",
        "system_failure": None,
    }
