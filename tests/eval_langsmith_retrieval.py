from dotenv import load_dotenv

from app.evaluation.retrieval_metrics import ndcg_at_k, precision_at_k, reciprocal_rank

load_dotenv(".env", override=True)

from pathlib import Path

from langsmith import Client

from app.rag.chunking.recursive import split_documents
from app.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingService,
)
from app.rag.loaders.langchain_loader import load_documents
from app.rag.retrieval.bm25_retriever import BM25Retriever
from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.retrieval.reranker import CrossEncoderReranker
from tests.eval_dataset import EVAL_DATASET

DATASET_NAME = "travel-retrieval-eval-v2"


def source_name(document) -> str:
    source = document.metadata.get("source", "")
    return Path(source).name


def build_dataset(client: Client) -> None:
    existing = list(
        client.list_datasets(
            dataset_name=DATASET_NAME,
        )
    )

    if existing:
        print("Dataset already exists.")
        return

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=("Golden retrieval dataset for agentic-travel-platform."),
    )

    for example in EVAL_DATASET:
        client.create_example(
            dataset_id=dataset.id,
            inputs={
                "query": example["query"],
            },
            outputs={
                "expected_source": (example["expected_source"]),
            },
        )

    print("Dataset created.")


def get_relevance_scores(
    retrieved_sources: list[str],
    expected_source: str,
) -> list[int]:
    return [1 if source == expected_source else 0 for source in retrieved_sources]


def retrieval_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> list[dict]:
    retrieved_sources = outputs.get(
        "sources",
        [],
    )

    expected_source = reference_outputs.get("expected_source")

    relevance_scores = get_relevance_scores(
        retrieved_sources,
        expected_source,
    )

    correct_rank = None

    for index, relevance in enumerate(
        relevance_scores,
        start=1,
    ):
        if relevance > 0:
            correct_rank = index
            break

    hit_at_3 = float(correct_rank is not None and correct_rank <= 3)

    mrr = reciprocal_rank(correct_rank)

    precision_3 = precision_at_k(
        relevance_scores,
        k=3,
    )

    ndcg_3 = ndcg_at_k(
        relevance_scores,
        k=3,
    )

    return [
        {
            "key": "hit_at_3",
            "score": hit_at_3,
        },
        {
            "key": "mrr",
            "score": mrr,
        },
        {
            "key": "precision_at_3",
            "score": precision_3,
        },
        {
            "key": "ndcg_at_3",
            "score": ndcg_3,
        },
    ]


def main() -> None:
    client = Client()

    build_dataset(client)

    documents = load_documents()

    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    embedding_service = SentenceTransformerEmbeddingService()

    dense_retriever = DenseRetriever(
        documents=chunks,
        embedding_service=embedding_service,
    )

    bm25_retriever = BM25Retriever(
        documents=chunks,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    reranker = CrossEncoderReranker()

    def dense_target(inputs: dict) -> dict:
        query = inputs["query"]

        results = dense_retriever.search(
            query=query,
            top_k=3,
        )

        return {"sources": [source_name(result) for result in results]}

    def hybrid_target(inputs: dict) -> dict:
        query = inputs["query"]

        results = hybrid_retriever.search(
            query=query,
            top_k=3,
        )

        return {"sources": [source_name(result) for result in results]}

    def hybrid_reranker_target(
        inputs: dict,
    ) -> dict:
        query = inputs["query"]

        candidates = hybrid_retriever.search(
            query=query,
            top_k=8,
        )

        results = reranker.rerank(
            query=query,
            documents=candidates,
            top_k=3,
        )

        return {"sources": [source_name(result) for result in results]}

    client.evaluate(
        dense_target,
        data=DATASET_NAME,
        evaluators=[
            retrieval_evaluator,
        ],
        experiment_prefix="dense-baseline",
        max_concurrency=1,
    )

    client.evaluate(
        hybrid_target,
        data=DATASET_NAME,
        evaluators=[
            retrieval_evaluator,
        ],
        experiment_prefix="hybrid-rrf",
        max_concurrency=1,
    )

    results = client.evaluate(
        hybrid_reranker_target,
        data=DATASET_NAME,
        evaluators=[
            retrieval_evaluator,
        ],
        experiment_prefix="hybrid-reranker",
        max_concurrency=1,
        blocking=True,
    )

    for result in results:
        print("OUTPUT:", result["run"].outputs)
        print("EVAL:", result["evaluation_results"]["results"])


if __name__ == "__main__":
    main()
