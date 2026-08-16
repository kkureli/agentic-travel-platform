from app.rag.chunking.recursive import (
    split_documents,
)
from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.loaders.langchain_loader import (
    load_documents,
)
from app.rag.retrieval.bm25_retriever import (
    BM25Retriever,
)
from app.rag.retrieval.dense_retriever import (
    DenseRetriever,
)
from app.rag.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from app.rag.retrieval.reranker import (
    CrossEncoderReranker,
)


def print_results(
    title: str,
    results,
) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        source = result.metadata.get(
            "source",
            "unknown",
        )

        print(f"{rank}. [{source}] {result.page_content[:150]}...")


def main() -> None:
    documents = load_documents()

    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    embedding_service = SentenceTransformerEmbeddingService()

    dense_retriever = DenseRetriever(
        documents=chunks,
        embedding_service=embedding_service,
    )

    bm25_retriever = BM25Retriever(
        documents=chunks,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    reranker = CrossEncoderReranker()

    query = "Can I get my hotel payment back if the reservation cannot be refunded?"

    candidates = hybrid_retriever.search(
        query=query,
        top_k=8,
        candidate_k=10,
    )

    reranked_results = reranker.rerank(
        query=query,
        documents=candidates,
        top_k=3,
    )

    print_results(
        "HYBRID CANDIDATES",
        candidates,
    )

    print_results(
        "AFTER CROSS-ENCODER RERANKING",
        reranked_results,
    )


if __name__ == "__main__":
    main()
