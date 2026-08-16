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
from app.rag.retrieval.multi_query_retriever import (
    MultiQueryRetriever,
)
from app.services.llm_service import (
    LLMService,
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

    llm_service = LLMService()

    multi_query_retriever = MultiQueryRetriever(
        llm_service=llm_service,
        retriever=hybrid_retriever,
        query_count=3,
    )

    query = "Can I get my money back?"

    generated_queries = multi_query_retriever.generate_queries(query)

    print("\nORIGINAL QUERY:")
    print(query)

    print("\nGENERATED QUERIES:")

    for generated_query in generated_queries:
        print(f"- {generated_query}")

    results = multi_query_retriever.search(
        query=query,
        top_k=5,
        candidate_k=8,
    )

    print("\nFINAL MULTI-QUERY RESULTS:")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        source = result.metadata.get(
            "source",
            "unknown",
        )

        print(f"{rank}. [{source}] {result.page_content[:150]}...")


if __name__ == "__main__":
    main()
