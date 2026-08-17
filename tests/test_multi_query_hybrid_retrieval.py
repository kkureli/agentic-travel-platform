from dotenv import load_dotenv

load_dotenv(
    ".env",
    override=True,
)

from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.retrieval.multi_query_generator import (
    MultiQueryGenerator,
)
from app.rag.retrieval.multi_query_hybrid_retriever import (
    MultiQueryHybridRetriever,
)
from app.rag.retrieval.qdrant_hybrid_retriever import (
    QdrantHybridRetriever,
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
        dense_embedding_service=(embedding_service),
    )

    reranker = CrossEncoderReranker()

    query_generator = MultiQueryGenerator(
        query_count=3,
    )

    retriever = MultiQueryHybridRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        query_generator=query_generator,
        query_count=3,
    )

    query = (
        "I paid for a hotel that says "
        "I can't get my money back. "
        "Is there any exception?"
    )

    results = retriever.search(
        query=query,
        tenant_id="travel-platform",
        top_k=3,
        per_query_k=10,
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
