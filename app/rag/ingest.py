"""Build the persistent dense index used by the hybrid retriever."""

from importlib import import_module

from app.config import settings
from app.rag.retriever import KB_DIR


def ingest() -> int:
    """Embed bundled KB files and upsert them into the configured Chroma collection."""
    paths = sorted(KB_DIR.glob("*.md"))
    if not paths:
        return 0
    documents = [path.read_text(encoding="utf-8") for path in paths]
    ids = [path.stem for path in paths]
    try:
        PersistentClient = import_module("chromadb").PersistentClient
        SentenceTransformer = import_module("sentence_transformers").SentenceTransformer
    except ModuleNotFoundError as error:
        raise RuntimeError("Install requirements.txt before building the knowledge-base index.") from error
    model = SentenceTransformer(settings.embedding_model)
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    client = PersistentClient(path=settings.vector_db_path)
    collection = client.get_or_create_collection("support_kb")
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=[{"source": path.name} for path in paths],
        embeddings=embeddings,
    )
    return len(documents)


if __name__ == "__main__":
    print(f"Indexed {ingest()} knowledge-base documents.")
