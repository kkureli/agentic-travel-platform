from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document

from app.rag.embeddings.embedding_service import EmbeddingService
from app.rag.retrieval.qdrant_hybrid_retriever import (
    HybridSearchResult,
    QdrantHybridRetriever,
)
from app.rag.retrieval.reranked_hybrid_retriever import (
    RerankedHybridRetriever,
)
from app.rag.retrieval.reranker import CrossEncoderReranker
from app.rag.vector_store.qdrant_store import QdrantStore

EVAL_PATH = Path("data/training/embedding_eval.jsonl")

FINETUNED_MODEL = "models/travel-embedding-triplet-v1/final"

TENANT_ID = "travel-platform"

K = 3

SearchResult = HybridSearchResult | Document


def load_eval_rows() -> list[dict]:
    with EVAL_PATH.open(
        encoding="utf-8",
    ) as file:
        return [json.loads(line) for line in file]


def calculate_metrics(
    ranked_ids: list[str],
    gold_ids: set[str],
    k: int,
) -> tuple[float, float, float]:
    top_k = ranked_ids[:k]

    # Recall@K
    relevant_retrieved = gold_ids.intersection(top_k)

    recall = len(relevant_retrieved) / len(gold_ids)

    # MRR
    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        ranked_ids,
        start=1,
    ):
        if chunk_id in gold_ids:
            reciprocal_rank = 1.0 / rank
            break

    # nDCG@K
    dcg = 0.0

    for rank, chunk_id in enumerate(
        top_k,
        start=1,
    ):
        if chunk_id in gold_ids:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(
        len(gold_ids),
        k,
    )

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1,
        )
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return (
        recall,
        reciprocal_rank,
        ndcg,
    )


def evaluate_retriever(
    name: str,
    retrieve,
    eval_rows: list[dict],
) -> dict[str, float]:
    recalls = []
    reciprocal_ranks = []
    ndcgs = []

    print(f"\n===== {name} =====")

    for row in eval_rows:
        query = row["query"]

        gold_ids = set(row["positive_chunk_ids"])

        results = retrieve(query)

        ranked_ids = [_chunk_id(result) for result in results]

        recall, mrr, ndcg = calculate_metrics(
            ranked_ids=ranked_ids,
            gold_ids=gold_ids,
            k=K,
        )

        recalls.append(recall)
        reciprocal_ranks.append(mrr)
        ndcgs.append(ndcg)

        print(f"\nQUERY: {query}")

        for rank, result in enumerate(
            results[:K],
            start=1,
        ):
            chunk_id = _chunk_id(result)

            marker = "✅" if chunk_id in gold_ids else "❌"

            source = _metadata(result).get(
                "source",
                "unknown",
            )

            score = getattr(result, "score", None)

            if score is None:
                print(f"{rank}. {marker} {source}")
            else:
                print(f"{rank}. {marker} {source} score={score:.4f}")

    return {
        "recall@3": float(np.mean(recalls)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "ndcg@3": float(np.mean(ndcgs)),
    }


def _metadata(result: SearchResult) -> dict:
    if isinstance(result, Document):
        return result.metadata

    return result.document.metadata


def _chunk_id(result: SearchResult) -> str:
    return str(_metadata(result)["chunk_id"])


def main() -> None:
    load_dotenv(
        ".env",
        override=True,
    )

    eval_rows = load_eval_rows()

    store = QdrantStore()

    embedding_service = EmbeddingService(
        model_name=FINETUNED_MODEL,
    )

    hybrid_retriever = QdrantHybridRetriever(
        store=store,
        dense_embedding_service=embedding_service,
    )

    reranked_retriever = RerankedHybridRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=CrossEncoderReranker(),
    )

    hybrid_metrics = evaluate_retriever(
        name="HYBRID + RRF",
        retrieve=lambda query: hybrid_retriever.search(
            query=query,
            tenant_id=TENANT_ID,
            top_k=10,
        ),
        eval_rows=eval_rows,
    )

    reranked_metrics = evaluate_retriever(
        name="HYBRID + CROSSENCODER",
        retrieve=lambda query: reranked_retriever.search(
            query=query,
            tenant_id=TENANT_ID,
            candidate_k=10,
            top_k=10,
        ),
        eval_rows=eval_rows,
    )

    print("\n===== FINAL RESULTS =====")

    print(
        "Hybrid + RRF:",
        hybrid_metrics,
    )

    print(
        "Hybrid + CrossEncoder:",
        reranked_metrics,
    )


if __name__ == "__main__":
    main()
