from __future__ import annotations

import uuid
from collections.abc import Sequence

from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
from qdrant_client import models

from app.rag.embeddings.embedding_service import EmbeddingService
from app.rag.vector_store.qdrant_store import QdrantStore

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

SPARSE_MODEL_NAME = "Qdrant/bm25"


class QdrantIngestionService:
    def __init__(
        self,
        store: QdrantStore,
        embedding_service: EmbeddingService,
    ) -> None:
        self.store = store
        self.embedding_service = embedding_service

        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    def ingest(
        self,
        documents: Sequence[Document],
        tenant_id: str,
        batch_size: int = 64,
    ) -> int:
        if not tenant_id:
            raise ValueError("tenant_id is required")

        if not documents:
            return 0

        total_indexed = 0

        for batch_start in range(
            0,
            len(documents),
            batch_size,
        ):
            batch = documents[batch_start : batch_start + batch_size]

            points = self._build_points(
                documents=batch,
                tenant_id=tenant_id,
            )

            self.store.client.upsert(
                collection_name=self.store.collection_name,
                points=points,
                wait=True,
            )

            total_indexed += len(points)

        return total_indexed

    def _build_points(
        self,
        documents: Sequence[Document],
        tenant_id: str,
    ) -> list[models.PointStruct]:
        texts = [document.page_content for document in documents]

        dense_vectors = self.embedding_service.embed_documents(texts)

        sparse_vectors = list(self.sparse_model.embed(texts))

        points: list[models.PointStruct] = []

        for (
            document,
            dense_vector,
            sparse_vector,
        ) in zip(
            documents,
            dense_vectors,
            sparse_vectors,
            strict=True,
        ):
            metadata = dict(document.metadata)

            source = str(metadata.get("source", "unknown"))

            document_id = metadata.get("document_id")

            if not document_id:
                document_id = self._build_document_id(
                    tenant_id=tenant_id,
                    source=source,
                )

            chunk_id = metadata.get("chunk_id")

            if not chunk_id:
                chunk_id = self._build_chunk_id(
                    document_id=document_id,
                    text=document.page_content,
                )

            point_id = self._build_point_id(
                tenant_id=tenant_id,
                chunk_id=chunk_id,
            )

            payload = {
                **metadata,
                "text": document.page_content,
                "tenant_id": tenant_id,
                "document_id": document_id,
                "chunk_id": chunk_id,
            }

            point = models.PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: list(dense_vector),
                    SPARSE_VECTOR_NAME: (
                        models.SparseVector(
                            indices=(sparse_vector.indices.tolist()),
                            values=(sparse_vector.values.tolist()),
                        )
                    ),
                },
                payload=payload,
            )

            points.append(point)

        return points

    @staticmethod
    def _build_document_id(
        tenant_id: str,
        source: str,
    ) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{tenant_id}:{source}",
            )
        )

    @staticmethod
    def _build_chunk_id(
        document_id: str,
        text: str,
    ) -> str:
        normalized_text = " ".join(text.split())

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{document_id}:{normalized_text}",
            )
        )

    @staticmethod
    def _build_point_id(
        tenant_id: str,
        chunk_id: str,
    ) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{tenant_id}:{chunk_id}",
            )
        )
