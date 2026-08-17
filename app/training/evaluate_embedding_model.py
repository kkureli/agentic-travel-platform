from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FINETUNED_MODEL = "models/travel-embedding-triplet-v1/final"
CATALOG_PATH = Path("data/training/retrieval_chunk_catalog.jsonl")

EVAL_PATH = Path("data/training/embedding_eval.jsonl")

K = 3


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def evaluate(
    model_name: str,
    catalog: list[dict],
    eval_rows: list[dict],
) -> dict[str, float]:
    model = SentenceTransformer(model_name)

    corpus_texts = [row["text"] for row in catalog]

    corpus_embeddings = model.encode(
        corpus_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    recalls = []
    reciprocal_ranks = []
    ndcgs = []

    for row in eval_rows:
        query = row["query"]

        gold_ids = set(row["positive_chunk_ids"])

        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = np.dot(
            corpus_embeddings,
            query_embedding,
        )

        ranked_indices = np.argsort(scores)[::-1]

        ranked_ids = [catalog[index]["chunk_id"] for index in ranked_indices]

        top_k = ranked_ids[:K]

        # Recall@K
        retrieved_relevant = gold_ids.intersection(top_k)

        recall = len(retrieved_relevant) / len(gold_ids)

        recalls.append(recall)

        # MRR
        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(
            ranked_ids,
            start=1,
        ):
            if chunk_id in gold_ids:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

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
            K,
        )

        idcg = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(
                1,
                ideal_hits + 1,
            )
        )

        ndcg = dcg / idcg if idcg > 0 else 0.0

        ndcgs.append(ndcg)

        print("\nQUERY:", query)

        for rank, index in enumerate(
            ranked_indices[:K],
            start=1,
        ):
            chunk = catalog[index]

            marker = "✅" if chunk["chunk_id"] in gold_ids else "❌"

            print(f"{rank}. {marker} {chunk['source']} score={scores[index]:.4f}")

    return {
        "recall@3": float(np.mean(recalls)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "ndcg@3": float(np.mean(ndcgs)),
    }


def main() -> None:
    catalog = load_jsonl(CATALOG_PATH)

    eval_rows = load_jsonl(EVAL_PATH)

    print("\n===== BASE MODEL =====")

    base_metrics = evaluate(
        BASE_MODEL,
        catalog,
        eval_rows,
    )

    print("\n===== FINE-TUNED MODEL =====")

    finetuned_metrics = evaluate(
        FINETUNED_MODEL,
        catalog,
        eval_rows,
    )

    print("\n===== RESULTS =====")

    print("Base:")
    print(base_metrics)

    print("\nFine-tuned:")
    print(finetuned_metrics)

    print("\nDelta:")

    for metric in base_metrics:
        delta = finetuned_metrics[metric] - base_metrics[metric]

        print(f"{metric}: {delta:+.4f}")


if __name__ == "__main__":
    main()
