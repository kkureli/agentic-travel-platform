from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INPUT_PATH = "data/training/embedding_seed.jsonl"
OUTPUT_PATH = "data/training/embedding_triplets.jsonl"


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    dataset = load_dataset(
        "json",
        data_files=INPUT_PATH,
        split="train",
    )

    hard_negative_dataset = mine_hard_negatives(
        dataset=dataset,
        model=model,
        anchor_column_name="query",
        positive_column_name="positive",
        relative_margin=0.05,
        range_min=1,
        range_max=10,
        min_score=0.30,
        num_negatives=2,
        sampling_strategy="top",
        output_format="triplet",
        output_scores=True,
        use_faiss=False,
    )

    hard_negative_dataset.to_json(
        OUTPUT_PATH,
        orient="records",
        lines=True,
    )

    print(f"Saved {len(hard_negative_dataset)} training examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
