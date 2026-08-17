from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

CATALOG_PATH = Path("data/training/retrieval_chunk_catalog.jsonl")

RELEVANCE_PATH = Path("data/training/embedding_relevance.jsonl")

OUTPUT_PATH = Path("data/training/embedding_finetuning_triplets.jsonl")

DENSE_CANDIDATE_K = 20
NUM_NEGATIVES = 2

RELATIVE_MARGIN = 0.05


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    source: str
    text: str


def load_catalog() -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []

    with CATALOG_PATH.open(
        encoding="utf-8",
    ) as file:
        for line in file:
            row = json.loads(line)

            chunks.append(
                CorpusChunk(
                    chunk_id=row["chunk_id"],
                    source=row["source"],
                    text=row["text"],
                )
            )

    return chunks


def load_relevance() -> list[dict]:
    rows: list[dict] = []

    with RELEVANCE_PATH.open(
        encoding="utf-8",
    ) as file:
        for line in file:
            rows.append(json.loads(line))

    return rows


def main() -> None:
    embedding_model = SentenceTransformer(MODEL_NAME)

    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    corpus = load_catalog()
    relevance_rows = load_relevance()

    chunk_by_id = {chunk.chunk_id: chunk for chunk in corpus}

    corpus_texts = [chunk.text for chunk in corpus]

    corpus_embeddings = embedding_model.encode(
        corpus_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    triplets: list[dict] = []

    for row in relevance_rows:
        query = row["query"]

        positive_ids = set(row["positive_chunk_ids"])

        positive_chunks = [chunk_by_id[chunk_id] for chunk_id in positive_ids]

        # ---------------------------------
        # 1. Dense candidate generation
        # ---------------------------------

        query_embedding = embedding_model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        dense_scores = np.dot(
            corpus_embeddings,
            query_embedding,
        )

        ranked_indices = np.argsort(dense_scores)[::-1]

        candidates: list[dict] = []

        for index in ranked_indices:
            chunk = corpus[index]

            # Gold positives can never be negatives.
            if chunk.chunk_id in positive_ids:
                continue

            candidates.append(
                {
                    "chunk": chunk,
                    "dense_score": float(dense_scores[index]),
                }
            )

            if len(candidates) >= DENSE_CANDIDATE_K:
                break

        # ---------------------------------
        # 2. Score gold positives
        # ---------------------------------

        positive_pairs = [
            (
                query,
                chunk.text,
            )
            for chunk in positive_chunks
        ]

        positive_scores = cross_encoder.predict(
            positive_pairs,
            show_progress_bar=False,
        )

        best_positive_score = float(max(positive_scores))

        # ---------------------------------
        # 3. CrossEncoder candidate scoring
        # ---------------------------------

        candidate_pairs = [
            (
                query,
                candidate["chunk"].text,
            )
            for candidate in candidates
        ]

        candidate_ce_scores = cross_encoder.predict(
            candidate_pairs,
            show_progress_bar=False,
        )

        scored_candidates: list[dict] = []

        for candidate, ce_score in zip(
            candidates,
            candidate_ce_scores,
            strict=True,
        ):
            ce_score = float(ce_score)

            # ---------------------------------
            # 4. False-negative protection
            # ---------------------------------

            margin = abs(best_positive_score) * RELATIVE_MARGIN

            max_allowed_negative_score = best_positive_score - margin

            if ce_score > max_allowed_negative_score:
                continue

            scored_candidates.append(
                {
                    "chunk": candidate["chunk"],
                    "dense_score": (candidate["dense_score"]),
                    "cross_encoder_score": (ce_score),
                }
            )

        # ---------------------------------
        # 5. Hardest valid negatives
        # ---------------------------------

        scored_candidates.sort(
            key=lambda item: item["cross_encoder_score"],
            reverse=True,
        )

        selected_negatives = scored_candidates[:NUM_NEGATIVES]

        # ---------------------------------
        # 6. Generate triplets
        # ---------------------------------

        for positive_chunk in positive_chunks:
            for negative_candidate in selected_negatives:
                negative_chunk = negative_candidate["chunk"]

                triplets.append(
                    {
                        "anchor": query,
                        "positive": (positive_chunk.text),
                        "negative": (negative_chunk.text),
                        "positive_chunk_id": (positive_chunk.chunk_id),
                        "negative_chunk_id": (negative_chunk.chunk_id),
                        "negative_dense_score": (negative_candidate["dense_score"]),
                        "negative_cross_encoder_score": (
                            negative_candidate["cross_encoder_score"]
                        ),
                    }
                )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for triplet in triplets:
            file.write(
                json.dumps(
                    triplet,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Generated {len(triplets)} fine-tuning triplets.")

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
