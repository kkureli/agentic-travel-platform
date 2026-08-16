from dotenv import load_dotenv

load_dotenv()
from app.evaluation.tracing import (
    trace_hybrid_retrieval,
    trace_reranking,
)
from app.rag.chunking.recursive import split_documents
from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.loaders.langchain_loader import load_documents
from app.rag.retrieval.bm25_retriever import BM25Retriever
from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.retrieval.reranker import CrossEncoderReranker


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

    query = "Can a non-refundable hotel reservation be refunded?"

    candidates = trace_hybrid_retrieval(
        retriever=hybrid_retriever,
        query=query,
        top_k=8,
    )

    final_results = trace_reranking(
        reranker=reranker,
        query=query,
        documents=candidates,
        top_k=3,
    )

    for rank, result in enumerate(
        final_results,
        start=1,
    ):
        print(
            rank,
            result.metadata.get("source"),
            result.page_content[:100],
        )


if __name__ == "__main__":
    main()
