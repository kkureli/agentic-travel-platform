from app.rag.chunking.recursive import split_documents
from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.loaders.langchain_loader import load_documents
from app.rag.retrieval.dense_retriever import DenseRetriever
from app.training.hard_negative_miner import HardNegativeMiner


def main() -> None:
    documents = load_documents()

    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    embedding_service = SentenceTransformerEmbeddingService()

    dense_retriever = DenseRetriever(
        documents=chunks,
        embedding_service=embedding_service,
    )

    miner = HardNegativeMiner(
        retriever=dense_retriever,
        candidate_k=5,
    )

    query = "Can a non-refundable hotel reservation be refunded?"

    positive = "Non-refundable reservations normally cannot be cancelled for a refund."

    hard_negative = miner.mine(
        query=query,
        positive=positive,
    )

    if hard_negative is None:
        print("No hard negative found.")
        return

    print("QUERY:")
    print(query)

    print("\nHARD NEGATIVE:")
    print(hard_negative.page_content)

    print("\nSOURCE:")
    print(hard_negative.metadata.get("source"))


if __name__ == "__main__":
    main()
