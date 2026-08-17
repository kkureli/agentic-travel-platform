from dataclasses import dataclass

from langchain_core.documents import Document

from app.rag.retrieval.multi_query_generator import (
    MultiQueryGenerator,
)
from app.rag.retrieval.qdrant_hybrid_retriever import (
    QdrantHybridRetriever,
)
from app.rag.retrieval.reranker import (
    CrossEncoderReranker,
)


@dataclass
class RankedDocument:
    document: Document
    score: float


class MultiQueryHybridRetriever:
    def __init__(
        self,
        hybrid_retriever: QdrantHybridRetriever,
        reranker: CrossEncoderReranker,
        query_generator: MultiQueryGenerator,
        query_count: int = 3,
        rrf_k: int = 60,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.query_generator = query_generator
        self.query_count = query_count
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 3,
        per_query_k: int = 10,
        language: str | None = None,
        document_type: str | None = None,
    ) -> list[Document]:

        alternative_queries = self.query_generator.generate(
            query=query,
            query_count=self.query_count,
        )

        queries = [
            query,
            *alternative_queries,
        ]

        fused_results: dict[
            str,
            RankedDocument,
        ] = {}

        for retrieval_query in queries:
            results = self.hybrid_retriever.search(
                query=retrieval_query,
                tenant_id=tenant_id,
                top_k=per_query_k,
                candidate_k=per_query_k,
                language=language,
                document_type=document_type,
            )

            for rank, result in enumerate(
                results,
                start=1,
            ):
                chunk_id = result.document.metadata.get("chunk_id")

                if not chunk_id:
                    continue

                rrf_score = 1.0 / (self.rrf_k + rank)

                existing = fused_results.get(chunk_id)

                if existing is None:
                    fused_results[chunk_id] = RankedDocument(
                        document=result.document,
                        score=rrf_score,
                    )

                else:
                    existing.score += rrf_score

        ranked_results = sorted(
            fused_results.values(),
            key=lambda item: item.score,
            reverse=True,
        )

        candidates = [ranked.document for ranked in ranked_results[:per_query_k]]

        if not candidates:
            return []

        return self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=top_k,
        )
