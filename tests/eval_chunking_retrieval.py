from langchain_core.documents import Document
from sentence_transformers import util

from app.evaluation.retrieval_metrics import (
    mean,
    recall_at_k,
    reciprocal_rank,
)
from app.rag.chunking.recursive import split_documents
from app.rag.chunking.semantic import SemanticChunker
from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.loaders.langchain_loader import load_documents
from tests.eval_dataset import EVAL_DATASET


def retrieve(
    query: str,
    chunks: list[Document],
    embedding_service: SentenceTransformerEmbeddingService,
    top_k: int = 3,
    metadata_filter: dict[str, str] | None = None,
) -> list[Document]:
    candidate_chunks = chunks

    if metadata_filter:
        candidate_chunks = [
            chunk
            for chunk in chunks
            if all(
                chunk.metadata.get(key) == value
                for key, value in metadata_filter.items()
            )
        ]

    if not candidate_chunks:
        return []

    query_embedding = embedding_service.model.encode(
        query,
        convert_to_tensor=True,
    )

    chunk_embeddings = embedding_service.model.encode(
        [chunk.page_content for chunk in candidate_chunks],
        convert_to_tensor=True,
    )

    similarities = util.cos_sim(
        query_embedding,
        chunk_embeddings,
    )[0]

    ranked_indices = similarities.argsort(
        descending=True,
    )

    return [candidate_chunks[index] for index in ranked_indices[:top_k]]


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
    strategy_name: str,
    chunks: list[Document],
    embedding_service: SentenceTransformerEmbeddingService,
    top_k: int = 3,
) -> None:
    recall_at_1_scores: list[float] = []
    recall_at_3_scores: list[float] = []
    reciprocal_rank_scores: list[float] = []

    print(f"\n{'=' * 60}")
    print(strategy_name)
    print(f"{'=' * 60}")

    for item in EVAL_DATASET:
        results = retrieve(
            query=item["query"],
            chunks=chunks,
            embedding_service=embedding_service,
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

        rr_score = reciprocal_rank(
            correct_rank=correct_rank,
        )

        recall_at_1_scores.append(recall_1)

        recall_at_3_scores.append(recall_3)

        reciprocal_rank_scores.append(rr_score)

        print(f"\nQuery: {item['query']}")

        print(f"Expected source: {item['expected_source']}")

        print(f"Correct rank: {correct_rank}")

        print(f"Recall@1: {recall_1:.3f}")

        print(f"Recall@3: {recall_3:.3f}")

        print(f"Reciprocal Rank: {rr_score:.3f}")

        print("\nRetrieved chunks:")

        for rank, result in enumerate(
            results,
            start=1,
        ):
            source = result.metadata.get(
                "source",
                "unknown",
            )

            print(f"{rank}. [{source}] {result.page_content[:120]}...")

    recall_1 = mean(recall_at_1_scores)

    recall_3 = mean(recall_at_3_scores)

    mrr = mean(reciprocal_rank_scores)

    print("\n--- FINAL METRICS ---")

    print(f"Recall@1: {recall_1:.3f}")

    print(f"Recall@3: {recall_3:.3f}")

    print(f"MRR: {mrr:.3f}")


def main() -> None:
    documents = load_documents("data/raw/knowledge_base")

    embedding_service = SentenceTransformerEmbeddingService()

    recursive_chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    semantic_chunker = SemanticChunker(
        embedding_service=embedding_service,
        minimum_similarity=0.30,
        max_chunk_size=500,
        chunk_overlap=100,
    )

    semantic_chunks: list[Document] = []

    for document in documents:
        semantic_chunks.extend(semantic_chunker.split_document(document))

    print(f"\nRecursive chunk count: {len(recursive_chunks)}")

    print(f"Semantic chunk count: {len(semantic_chunks)}")

    evaluate(
        strategy_name="RECURSIVE CHUNKING",
        chunks=recursive_chunks,
        embedding_service=embedding_service,
        top_k=3,
    )

    evaluate(
        strategy_name="SEMANTIC CHUNKING",
        chunks=semantic_chunks,
        embedding_service=embedding_service,
        top_k=3,
    )


if __name__ == "__main__":
    main()
