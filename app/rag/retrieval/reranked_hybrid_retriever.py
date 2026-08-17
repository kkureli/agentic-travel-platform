from langchain_core.documents import Document

from app.rag.retrieval.qdrant_hybrid_retriever import (
    QdrantHybridRetriever,
)
from app.rag.retrieval.reranker import (
    CrossEncoderReranker,
)


class RerankedHybridRetriever:
    def __init__(
        self,
        hybrid_retriever: QdrantHybridRetriever,
        reranker: CrossEncoderReranker,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 3,
        candidate_k: int = 20,
        language: str | None = None,
        document_type: str | None = None,
    ) -> list[Document]:
        hybrid_results = self.hybrid_retriever.search(
            query=query,
            tenant_id=tenant_id,
            top_k=candidate_k,
            candidate_k=candidate_k,
            language=language,
            document_type=document_type,
        )

        candidates = [result.document for result in hybrid_results]

        reranked = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=top_k,
        )

        return reranked
