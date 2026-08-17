from datasets import load_dataset
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)
from sentence_transformers.util import (
    mine_hard_negatives,
)

from app.rag.ingestion.retrieval_document_builder import (
    RetrievalDocumentBuilder,
)
from app.rag.loaders.langchain_loader import (
    load_documents,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

SEED_DATASET_PATH = "data/training/embedding_seed.jsonl"

OUTPUT_PATH = "data/training/embedding_triplets.jsonl"


def build_corpus() -> list[str]:
    documents = load_documents()

    builder = RetrievalDocumentBuilder()

    chunks = builder.build(documents)

    return [chunk.page_content for chunk in chunks]


def main() -> None:
    embedding_model = SentenceTransformer(MODEL_NAME)

    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    seed_dataset = load_dataset(
        "json",
        data_files=SEED_DATASET_PATH,
        split="train",
    )

    corpus = build_corpus()

    print(f"Seed examples: {len(seed_dataset)}")

    print(f"Corpus chunks: {len(corpus)}")

    triplet_dataset = mine_hard_negatives(
        dataset=seed_dataset,
        model=embedding_model,
        corpus=corpus,
        anchor_column_name="query",
        positive_column_name="positive",
        cross_encoder=cross_encoder,
        relative_margin=0.05,
        range_min=1,
        range_max=20,
        min_score=None,
        num_negatives=2,
        sampling_strategy="top",
        output_format="triplet",
        output_scores=True,
        use_faiss=False,
    )

    triplet_dataset.to_json(
        OUTPUT_PATH,
        orient="records",
        lines=True,
    )

    print(f"Generated {len(triplet_dataset)} triplets.")

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
