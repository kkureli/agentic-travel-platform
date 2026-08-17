from dataclasses import dataclass

from langchain_core.documents import Document

from app.rag.retrieval.qdrant_dense_retriever import (
    QdrantDenseRetriever,
)


@dataclass
class HardNegative:
    document: Document
    score: float


class HardNegativeMiner:
    def __init__(
        self,
        retriever: QdrantDenseRetriever,
        candidate_k: int = 10,
        min_similarity: float = 0.35,
        relative_margin: float = 0.05,
    ):
        self.retriever = retriever
        self.candidate_k = candidate_k
        self.min_similarity = min_similarity
        self.relative_margin = relative_margin

    def mine(
        self,
        query: str,
        tenant_id: str,
        positive_chunk_id: str,
    ) -> HardNegative | None:
        candidates = self.retriever.search(
            query=query,
            tenant_id=tenant_id,
            top_k=self.candidate_k,
        )

        positive_result = next(
            (
                result
                for result in candidates
                if result.document.metadata.get("chunk_id") == positive_chunk_id
            ),
            None,
        )

        if positive_result is None:
            return None

        positive_score = positive_result.score

        max_negative_score = positive_score - abs(positive_score) * self.relative_margin

        for result in candidates:
            chunk_id = result.document.metadata.get("chunk_id")

            if chunk_id == positive_chunk_id:
                continue

            if result.score > max_negative_score:
                continue

            if result.score < self.min_similarity:
                continue

            return HardNegative(
                document=result.document,
                score=result.score,
            )

        return None
