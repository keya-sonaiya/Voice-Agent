"""Hybrid dense and BM25 retrieval for the bundled support knowledge base."""

from collections.abc import Sequence
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from string import punctuation

from app.config import settings

KB_DIR = Path(__file__).parent / "kb"


def _tokenize(text: str) -> list[str]:
    return [word.strip(punctuation).lower() for word in text.split() if word.strip(punctuation)]


@lru_cache(maxsize=1)
def _documents() -> tuple[tuple[str, str], ...]:
    """Load the small, auditable source corpus once per process."""
    return tuple((path.name, path.read_text(encoding="utf-8")) for path in sorted(KB_DIR.glob("*.md")))


def _dense_documents(query: str, limit: int) -> list[str]:
    """Fetch dense matches from Chroma, with a local embedding fallback before ingest runs."""
    try:
        PersistentClient = import_module("chromadb").PersistentClient
        SentenceTransformer = import_module("sentence_transformers").SentenceTransformer
        client = PersistentClient(path=settings.vector_db_path)
        collection = client.get_collection("support_kb")
        model = SentenceTransformer(settings.embedding_model)
        query_embedding = model.encode([query], normalize_embeddings=True).tolist()
        result = collection.query(query_embeddings=query_embedding, n_results=limit, include=["documents"])
        documents = result.get("documents", [[]])
        if documents and documents[0]:
            return [str(item) for item in documents[0]]
    except Exception:
        # The index may not exist during first-run setup; use the same configured model in memory.
        pass
    corpus = [text for _, text in _documents()]
    if not corpus:
        return []
    try:
        SentenceTransformer = import_module("sentence_transformers").SentenceTransformer
    except ModuleNotFoundError as error:
        raise RuntimeError("Install requirements.txt before using hybrid retrieval.") from error
    model = SentenceTransformer(settings.embedding_model)
    vectors = model.encode([query, *corpus], normalize_embeddings=True)
    query_vector = vectors[0]
    scores = [(float(query_vector @ vector), text) for vector, text in zip(vectors[1:], corpus, strict=True)]
    return [text for _, text in sorted(scores, reverse=True)[:limit]]


def _rrf_rank(result_sets: Sequence[Sequence[str]], limit: int) -> list[str]:
    """Fuse ranked dense and lexical lists with deterministic reciprocal-rank fusion."""
    scores: dict[str, float] = {}
    for result_set in result_sets:
        for rank, document in enumerate(result_set, start=1):
            scores[document] = scores.get(document, 0.0) + 1 / (60 + rank)
    return [document for document, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]


def retrieve(query: str, limit: int = 3) -> list[str]:
    """Return source-labelled excerpts ranked by both dense similarity and BM25 relevance."""
    source_documents = _documents()
    if not source_documents:
        return []
    source_by_text = {text: f"Source: {name}\n{text}" for name, text in source_documents}
    corpus = [text for _, text in source_documents]
    try:
        BM25Okapi = import_module("rank_bm25").BM25Okapi
    except ModuleNotFoundError as error:
        raise RuntimeError("Install requirements.txt before using hybrid retrieval.") from error
    bm25 = BM25Okapi([_tokenize(text) for text in corpus])
    lexical = [
        text
        for _, text in sorted(
            zip(bm25.get_scores(_tokenize(query)), corpus, strict=True), key=lambda item: item[0], reverse=True
        )[:limit]
    ]
    dense = _dense_documents(query, limit)
    fused = _rrf_rank([dense, lexical], limit)
    return [source_by_text.get(document, f"Source: indexed document\n{document}") for document in fused]
