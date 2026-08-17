from qdrant_client import QdrantClient, models

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

COLLECTION_NAME = "travel_knowledge"

DENSE_VECTOR_SIZE = 384


class QdrantStore:
    def __init__(
        self,
        url: str = "http://localhost:6337",
        collection_name: str = COLLECTION_NAME,
    ):
        self.client = QdrantClient(
            url=url,
        )

        self.collection_name = collection_name

    def recreate_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )

        self._create_payload_indexes()

    def _create_payload_indexes(
        self,
    ) -> None:
        keyword_fields = [
            "tenant_id",
            "document_id",
            "chunk_id",
            "source",
            "language",
            "document_type",
        ]

        for field in keyword_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=(models.PayloadSchemaType.KEYWORD),
            )
