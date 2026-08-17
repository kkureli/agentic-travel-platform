from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.retrieval.qdrant_hybrid_retriever import (
    QdrantHybridRetriever,
)
from app.rag.vector_store.qdrant_store import (
    QdrantStore,
)


def main() -> None:
    store = QdrantStore()

    embedding_service = SentenceTransformerEmbeddingService()

    retriever = QdrantHybridRetriever(
        store=store,
        dense_embedding_service=embedding_service,
    )

    results = retriever.search(
        query=("Can a non-refundable hotel reservation be refunded?"),
        tenant_id="travel-platform",
        top_k=5,
        candidate_k=10,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(f"\nRANK: {rank}")

        print(f"SCORE: {result.score:.4f}")

        print(
            "SOURCE:",
            result.document.metadata.get("source"),
        )

        print(
            "CHUNK ID:",
            result.document.metadata.get("chunk_id"),
        )

        print(result.document.page_content[:200])


if __name__ == "__main__":
    main()
