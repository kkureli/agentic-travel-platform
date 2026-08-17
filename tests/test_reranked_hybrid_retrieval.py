from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.retrieval.qdrant_hybrid_retriever import (
    QdrantHybridRetriever,
)
from app.rag.retrieval.reranked_hybrid_retriever import (
    RerankedHybridRetriever,
)
from app.rag.retrieval.reranker import (
    CrossEncoderReranker,
)
from app.rag.vector_store.qdrant_store import (
    QdrantStore,
)


def main() -> None:
    store = QdrantStore()

    embedding_service = SentenceTransformerEmbeddingService()

    hybrid_retriever = QdrantHybridRetriever(
        store=store,
        dense_embedding_service=embedding_service,
    )

    reranker = CrossEncoderReranker()

    retriever = RerankedHybridRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )

    query = "Can a non-refundable hotel reservation be refunded?"

    results = retriever.search(
        query=query,
        tenant_id="travel-platform",
        candidate_k=10,
        top_k=3,
    )

    for rank, document in enumerate(
        results,
        start=1,
    ):
        print(f"\nRANK: {rank}")

        print(
            "SOURCE:",
            document.metadata.get("source"),
        )

        print(
            "CHUNK ID:",
            document.metadata.get("chunk_id"),
        )

        print(document.page_content[:300])


if __name__ == "__main__":
    main()
