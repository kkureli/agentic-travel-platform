from dotenv import load_dotenv

load_dotenv(".env", override=True)

from app.rag.embeddings.embedding_service import (
    EmbeddingService,
)
from app.rag.retrieval.qdrant_hybrid_retriever import (
    QdrantHybridRetriever,
)
from app.rag.vector_store.qdrant_store import (
    QdrantStore,
)
from app.tools.travel_rag_tool import (
    build_travel_rag_tool,
)

EMBEDDING_MODEL = "models/travel-embedding-triplet-v1/final"


def main() -> None:
    store = QdrantStore()

    embedding_service = EmbeddingService(
        model_name=EMBEDDING_MODEL,
    )

    retriever = QdrantHybridRetriever(
        store=store,
        dense_embedding_service=embedding_service,
    )

    rag_tool = build_travel_rag_tool(retriever)

    result = rag_tool.invoke(
        {"query": ("Can a non-refundable hotel reservation be refunded?")}
    )

    print(result)


if __name__ == "__main__":
    main()
