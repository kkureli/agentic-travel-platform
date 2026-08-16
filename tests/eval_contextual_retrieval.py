from langchain_core.documents import Document

from app.evaluation.retrieval_metrics import (
    mean,
    recall_at_k,
    reciprocal_rank,
)
from app.rag.chunking.recursive import split_documents
from app.rag.contextual_ingestion import (
    build_contextual_chunks,
)
from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.loaders.langchain_loader import (
    load_documents,
)
from app.rag.retrieval.dense_retriever import (
    DenseRetriever,
)
from tests.eval_dataset import EVAL_DATASET


def find_correct_rank(
    results: list[Document],
    expected_source: str,
) -> int | None:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        source = result.metadata.get(
            "source",
            "",
        )

        if source.endswith(expected_source):
            return rank

    return None


def evaluate(
    name: str,
    retriever: DenseRetriever,
    top_k: int = 3,
) -> None:
    recall_1_scores: list[float] = []
    recall_3_scores: list[float] = []
    rr_scores: list[float] = []

    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    for item in EVAL_DATASET:
        results = retriever.search(
            query=item["query"],
            top_k=top_k,
        )

        correct_rank = find_correct_rank(
            results=results,
            expected_source=item["expected_source"],
        )

        recall_1 = recall_at_k(
            correct_rank=correct_rank,
            k=1,
        )

        recall_3 = recall_at_k(
            correct_rank=correct_rank,
            k=3,
        )

        rr = reciprocal_rank(
            correct_rank=correct_rank,
        )

        recall_1_scores.append(recall_1)
        recall_3_scores.append(recall_3)
        rr_scores.append(rr)

        print(f"\nQuery: {item['query']}")

        print(f"Correct rank: {correct_rank}")

        for rank, result in enumerate(
            results,
            start=1,
        ):
            source = result.metadata.get(
                "source",
                "unknown",
            )

            print(f"{rank}. [{source}] {result.page_content[:120]}...")

    print("\n--- FINAL METRICS ---")

    print(f"Recall@1: {mean(recall_1_scores):.3f}")

    print(f"Recall@3: {mean(recall_3_scores):.3f}")

    print(f"MRR: {mean(rr_scores):.3f}")


def main() -> None:
    embedding_service = SentenceTransformerEmbeddingService()

    documents = load_documents()

    normal_chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    contextual_chunks = build_contextual_chunks(
        chunk_size=500,
        chunk_overlap=100,
    )

    normal_retriever = DenseRetriever(
        documents=normal_chunks,
        embedding_service=embedding_service,
    )

    contextual_retriever = DenseRetriever(
        documents=contextual_chunks,
        embedding_service=embedding_service,
    )

    print(f"\nNormal chunks: {len(normal_chunks)}")

    print(f"Contextual chunks: {len(contextual_chunks)}")

    evaluate(
        name="NORMAL RETRIEVAL",
        retriever=normal_retriever,
        top_k=3,
    )

    evaluate(
        name="CONTEXTUAL RETRIEVAL",
        retriever=contextual_retriever,
        top_k=3,
    )


if __name__ == "__main__":
    main()
