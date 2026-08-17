from dataclasses import dataclass

from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
from qdrant_client import models

from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.vector_store.qdrant_store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantStore,
)


@dataclass
class HybridSearchResult:
    document: Document
    score: float
    point_id: str


class QdrantHybridRetriever:
    def __init__(
        self,
        store: QdrantStore,
        dense_embedding_service: SentenceTransformerEmbeddingService,
    ):
        self.store = store
        self.dense_embedding_service = dense_embedding_service

        self.sparse_embedding_service = SparseTextEmbedding(
            model_name="Qdrant/bm25",
        )

    def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 10,
        candidate_k: int = 20,
        language: str | None = None,
        document_type: str | None = None,
    ) -> list[HybridSearchResult]:
        dense_query = self.dense_embedding_service.embed_query(query)

        sparse_query = next(iter(self.sparse_embedding_service.query_embed(query)))

        query_filter = self._build_filter(
            tenant_id=tenant_id,
            language=language,
            document_type=document_type,
        )

        response = self.store.client.query_points(
            collection_name=self.store.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using=DENSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=candidate_k,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist(),
                    ),
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=candidate_k,
                ),
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        results: list[HybridSearchResult] = []

        for point in response.points:
            payload = point.payload or {}

            document = Document(
                page_content=payload.get(
                    "text",
                    "",
                ),
                metadata={
                    key: value for key, value in payload.items() if key != "text"
                },
            )

            results.append(
                HybridSearchResult(
                    document=document,
                    score=float(point.score),
                    point_id=str(point.id),
                )
            )

        return results

    def _build_filter(
        self,
        tenant_id: str,
        language: str | None,
        document_type: str | None,
    ) -> models.Filter:
        conditions: list[models.FieldCondition] = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(
                    value=tenant_id,
                ),
            )
        ]

        if language is not None:
            conditions.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(
                        value=language,
                    ),
                )
            )

        if document_type is not None:
            conditions.append(
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(
                        value=document_type,
                    ),
                )
            )

        return models.Filter(
            must=conditions,
        )
