from dotenv import load_dotenv

from app.rag.embeddings.embedding_service import (
    EmbeddingService,
)
from app.rag.ingestion.qdrant_ingestion import (
    QdrantIngestionService,
)
from app.rag.ingestion.retrieval_document_builder import (
    RetrievalDocumentBuilder,
)
from app.rag.loaders.langchain_loader import (
    load_documents,
)
from app.rag.vector_store.qdrant_store import (
    QdrantStore,
)

TENANT_ID = "travel-platform"


def main() -> None:
    load_dotenv(
        ".env",
        override=True,
    )

    print("Loading knowledge base...")

    raw_documents = load_documents()

    print(f"Loaded {len(raw_documents)} documents.")

    builder = RetrievalDocumentBuilder()

    retrieval_documents = builder.build(raw_documents)

    print(f"Generated {len(retrieval_documents)} retrieval chunks.")

    embedding_service = EmbeddingService(
        model_name=("models/travel-embedding-triplet-v1/final")
    )
    store = QdrantStore()

    ingestion_service = QdrantIngestionService(
        store=store,
        embedding_service=embedding_service,
    )

    print(f"Recreating Qdrant collection: {store.collection_name}")

    store.recreate_collection()

    print("Indexing retrieval chunks...")

    indexed_count = ingestion_service.ingest(
        documents=retrieval_documents,
        tenant_id=TENANT_ID,
    )

    print(f"Indexed {indexed_count} chunks into {store.collection_name}.")


if __name__ == "__main__":
    main()
