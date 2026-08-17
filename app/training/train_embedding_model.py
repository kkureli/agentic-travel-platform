from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import (
    TripletDistanceMetric,
    TripletLoss,
)

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TRAIN_DATASET_PATH = "data/training/embedding_finetuning_triplets.jsonl"

OUTPUT_DIR = Path("models/travel-embedding-triplet-v1")


def main() -> None:
    model = SentenceTransformer(BASE_MODEL_NAME)

    dataset = load_dataset(
        "json",
        data_files=TRAIN_DATASET_PATH,
        split="train",
    )

    train_dataset = dataset.select_columns(
        [
            "anchor",
            "positive",
            "negative",
        ]
    )

    loss = TripletLoss(
        model=model,
        distance_metric=(TripletDistanceMetric.COSINE),
        triplet_margin=0.2,
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=1,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=42,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    print(f"Training examples: {len(train_dataset)}")

    trainer.train()

    final_model_path = OUTPUT_DIR / "final"

    model.save_pretrained(str(final_model_path))

    print(f"Fine-tuned model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
