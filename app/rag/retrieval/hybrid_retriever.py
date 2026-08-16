from langchain_core.documents import Document

from app.rag.retrieval.bm25_retriever import (
    BM25Retriever,
)
from app.rag.retrieval.dense_retriever import (
    DenseRetriever,
)
from app.rag.retrieval.rrf import (
    reciprocal_rank_fusion,
)


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever

    def search(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 5,
    ) -> list[Document]:
        dense_results = self.dense_retriever.search(
            query=query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=candidate_k,
        )

        return reciprocal_rank_fusion(
            rankings=[
                dense_results,
                bm25_results,
            ],
            top_k=top_k,
        )
