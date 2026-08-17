from __future__ import annotations

import json
import re
from pathlib import Path

INPUT_PATH = Path("data/training/embedding_triplets.jsonl")

CLEAN_OUTPUT_PATH = Path("data/training/embedding_triplets_clean.jsonl")

REVIEW_OUTPUT_PATH = Path("data/training/embedding_triplets_review.jsonl")

REJECTED_OUTPUT_PATH = Path("data/training/embedding_triplets_rejected.jsonl")

# CrossEncoder positive-negative score farkı bundan küçükse
# otomatik silmek yerine manual review'e gönderiyoruz.
REVIEW_SCORE_GAP = 2.0


def normalize_text(text: str) -> str:
    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    return text.strip()


def contains_positive(
    positive: str,
    negative: str,
) -> bool:
    positive_normalized = normalize_text(positive)

    negative_normalized = normalize_text(negative)

    return positive_normalized in negative_normalized


def get_score_gap(
    scores: list[float],
) -> float:
    positive_score = scores[0]
    negative_score = scores[1]

    return positive_score - negative_score


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    clean: list[dict] = []
    review: list[dict] = []
    rejected: list[dict] = []

    with INPUT_PATH.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)

            query = row["query"]
            positive = row["positive"]
            negative = row["negative"]
            scores = row["scores"]

            score_gap = get_score_gap(scores)

            # Strong deterministic evidence:
            # negative directly contains the gold answer.
            if contains_positive(
                positive,
                negative,
            ):
                row["validation_reason"] = "negative_contains_positive"

                rejected.append(row)
                continue

            # CrossEncoder thinks positive and negative
            # are suspiciously close.
            if score_gap < REVIEW_SCORE_GAP:
                row["validation_reason"] = "small_cross_encoder_gap"

                row["score_gap"] = score_gap

                review.append(row)
                continue

            clean.append(row)

    write_jsonl(
        CLEAN_OUTPUT_PATH,
        clean,
    )

    write_jsonl(
        REVIEW_OUTPUT_PATH,
        review,
    )

    write_jsonl(
        REJECTED_OUTPUT_PATH,
        rejected,
    )

    print(f"Input: {len(clean) + len(review) + len(rejected)}")

    print(f"Clean: {len(clean)}")

    print(f"Review: {len(review)}")

    print(f"Rejected: {len(rejected)}")

    print(f"Clean dataset: {CLEAN_OUTPUT_PATH}")

    print(f"Review dataset: {REVIEW_OUTPUT_PATH}")

    print(f"Rejected dataset: {REJECTED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
