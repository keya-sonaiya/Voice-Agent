"""Deterministic rolling sentiment tracker."""

from app.graph.state import ConversationState

_NEGATIVE = {
    "angry",
    "awful",
    "cancel",
    "disgusting",
    "frustrated",
    "hate",
    "terrible",
    "useless",
    "worst",
}
_POSITIVE = {"appreciate", "great", "helpful", "thanks", "thank", "wonderful"}


def _score(text: str) -> float:
    words = {word.strip(".,!?;:").lower() for word in text.split()}
    return max(-1.0, min(1.0, (len(words & _POSITIVE) - len(words & _NEGATIVE)) / 2))


def update_sentiment(state: ConversationState) -> dict[str, object]:
    """Read `current_transcript`/`rolling_sentiment`/`turns`; write sentiment and caller turn; never invoke an LLM."""
    score = _score(state["current_transcript"])
    # Equal weighting is intentionally simple and keeps the score bounded and auditable.
    rolling = (state["rolling_sentiment"] + score) / 2
    last_turn = state["turns"][-1].model_copy(update={"sentiment_score": score})
    turns = [*state["turns"][:-1], last_turn]
    return {"rolling_sentiment": rolling, "turns": turns}
