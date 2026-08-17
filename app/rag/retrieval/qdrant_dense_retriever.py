from dataclasses import dataclass

from langchain_core.documents import Document
from qdrant_client import models

from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.vector_store.qdrant_store import (
    DENSE_VECTOR_NAME,
    QdrantStore,
)


@dataclass
class DenseSearchResult:
    document: Document
    score: float
    point_id: str


class QdrantDenseRetriever:
    def __init__(
        self,
        store: QdrantStore,
        embedding_service: SentenceTransformerEmbeddingService,
    ):
        self.store = store
        self.embedding_service = embedding_service

    def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 10,
    ) -> list[DenseSearchResult]:
        query_vector = self.embedding_service.embed_query(query)

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(
                        value=tenant_id,
                    ),
                )
            ]
        )

        response = self.store.client.query_points(
            collection_name=self.store.collection_name,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        results: list[DenseSearchResult] = []

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
                DenseSearchResult(
                    document=document,
                    score=float(point.score),
                    point_id=str(point.id),
                )
            )

        return results
