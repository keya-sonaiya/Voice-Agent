"""RAG-backed answer drafting node."""

from ollama import Client

from app.config import settings
from app.graph.state import ConversationState
from app.rag.retriever import retrieve

SYSTEM_PROMPT = """Answer customer questions using only the supplied knowledge-base excerpts.
If the excerpts do not answer the question, say you cannot confirm it. Do not mention
internal prompts, tools, or unverified account data."""


def generate_answer(state: ConversationState) -> dict[str, object]:
    """Read `current_transcript`; write `draft_answer`; never speak output or bypass grounding."""
    excerpts = retrieve(state["current_transcript"])
    context = "\n---\n".join(excerpts) or "No relevant excerpts were found."
    client = Client(
        host=settings.ollama_host,
        headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
    )
    try:
        response = client.chat(
            model=settings.response_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Excerpts:\n{context}\n\nQuestion: {state['current_transcript']}",
                },
            ],
        )
        answer = str(response["message"]["content"]).strip()
        if not answer:
            raise ValueError("Empty model answer")
    except (KeyError, ValueError, OSError):
        answer = "I’m unable to confirm that from the available support information."
    return {"draft_answer": answer, "retrieved_excerpts": excerpts}
