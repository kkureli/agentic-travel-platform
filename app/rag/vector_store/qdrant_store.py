from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from app.core.config import get_settings

COLLECTION_NAME = "travel_knowledge"
VECTOR_SIZE = 384


class QdrantVectorStore:
    def __init__(self):
        settings = get_settings()

        self.client = QdrantClient(
            url=settings.qdrant_url,
        )

    def create_collection(self) -> None:
        if self.client.collection_exists(COLLECTION_NAME):
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    def index_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        points = []

        for index, (document, embedding) in enumerate(zip(documents, embeddings)):
            point = models.PointStruct(
                id=index,
                vector=embedding,
                payload={
                    "text": document.page_content,
                    "metadata": document.metadata,
                },
            )

            points.append(point)

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 3,
    ):
        result = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return result.points
