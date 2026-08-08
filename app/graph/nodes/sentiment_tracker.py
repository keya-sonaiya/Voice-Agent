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
_NEGATORS = {"not", "no", "never", "don't", "doesn't", "isn't", "won't"}


def _score(text: str) -> float:
    words = [word.strip(".,!?;:").lower() for word in text.split()]
    total = 0
    for index, word in enumerate(words):
        negated = index > 0 and words[index - 1] in _NEGATORS
        if word in _POSITIVE:
            total += -1 if negated else 1
        elif word in _NEGATIVE:
            total += 1 if negated else -1
    return max(-1.0, min(1.0, total * 4 / max(len(words), 1)))


def update_sentiment(state: ConversationState) -> dict[str, object]:
    """Read `current_transcript`/`rolling_sentiment`/`turns`; write sentiment and caller turn; never invoke an LLM."""
    score = _score(state["current_transcript"])
    # Equal weighting is intentionally simple and keeps the score bounded and auditable.
    rolling = (state["rolling_sentiment"] + score) / 2
    last_turn = state["turns"][-1].model_copy(update={"sentiment_score": score})
    turns = [*state["turns"][:-1], last_turn]
    return {"rolling_sentiment": rolling, "turns": turns}
