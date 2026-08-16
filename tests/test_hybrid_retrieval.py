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

    queries = [
        "non-refundable reservation",
        "Belek family hotels",
        "loyalty program points",
        ("Can I get my hotel payment back if I cannot cancel?"),
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"QUERY: {query}")
        print(f"{'=' * 60}")

        print("\nDENSE:")

        dense_results = dense_retriever.search(
            query=query,
            top_k=3,
        )

        _print_results(dense_results)

        print("\nBM25:")

        bm25_results = bm25_retriever.search(
            query=query,
            top_k=3,
        )

        _print_results(bm25_results)

        print("\nHYBRID:")

        hybrid_results = hybrid_retriever.search(
            query=query,
            top_k=3,
            candidate_k=5,
        )

        _print_results(hybrid_results)


def _print_results(
    results,
) -> None:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        source = result.metadata.get(
            "source",
            "unknown",
        )

        print(f"{rank}. [{source}] {result.page_content[:120]}...")


if __name__ == "__main__":
    main()
