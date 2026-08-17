from __future__ import annotations

import json
from pathlib import Path

CATALOG_PATH = Path("data/training/retrieval_chunk_catalog.jsonl")

LABELS_PATH = Path("data/training/embedding_training_labels.jsonl")

OUTPUT_PATH = Path("data/training/embedding_finetuning_triplets.jsonl")


def load_catalog() -> dict[str, dict]:
    catalog: dict[str, dict] = {}

    with CATALOG_PATH.open(
        encoding="utf-8",
    ) as file:
        for line in file:
            row = json.loads(line)

            catalog[row["chunk_id"]] = row

    return catalog


def load_labels() -> list[dict]:
    rows: list[dict] = []

    with LABELS_PATH.open(
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def main() -> None:
    catalog = load_catalog()
    labels = load_labels()

    triplets: list[dict] = []

    for row in labels:
        query = row["query"]

        positive_ids = row["positive_chunk_ids"]

        negative_ids = row["negative_chunk_ids"]

        for positive_id in positive_ids:
            positive = catalog[positive_id]["text"]

            for negative_id in negative_ids:
                negative = catalog[negative_id]["text"]

                triplets.append(
                    {
                        "anchor": query,
                        "positive": positive,
                        "negative": negative,
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

    print(f"Generated {len(triplets)} training triplets.")

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
