import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client, evaluate
from pydantic import BaseModel, Field

load_dotenv(".env", override=True)


from app.workflows.agentic_rag.graph import (
    build_agentic_rag_graph,
)

DATASET_NAME = "agentic-rag-workflow-eval-v2"

EVAL_FILE = Path("data/evaluation/agentic_rag_eval.jsonl")


graph = build_agentic_rag_graph()


judge_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)


class CorrectnessGrade(BaseModel):
    score: float = Field(
        ge=0,
        le=1,
        description=("0 means incorrect and 1 means fully correct."),
    )

    reason: str = Field(description=("A concise explanation for the assigned score."))


class GroundednessGrade(BaseModel):
    score: float = Field(
        ge=0,
        le=1,
        description=(
            "0 means unsupported by the evidence and 1 means fully supported."
        ),
    )

    reason: str = Field(description=("A concise explanation for the assigned score."))


correctness_judge = judge_llm.with_structured_output(CorrectnessGrade)


groundedness_judge = judge_llm.with_structured_output(GroundednessGrade)


def load_examples() -> list[dict]:
    examples: list[dict] = []

    with EVAL_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            examples.append(json.loads(line))

    return examples


def create_dataset_if_needed(
    client: Client,
) -> None:
    existing_datasets = list(
        client.list_datasets(
            dataset_name=DATASET_NAME,
        )
    )

    if existing_datasets:
        print(f"Dataset already exists: {DATASET_NAME}")
        return

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Evaluation dataset for the "
            "agentic RAG workflow including "
            "routing, status, retry behavior, "
            "answer correctness, and groundedness."
        ),
    )

    examples = load_examples()

    langsmith_examples = []

    for example in examples:
        langsmith_examples.append(
            {
                "inputs": {
                    "query": example["query"],
                },
                "outputs": {
                    "expected_route": (example["expected_route"]),
                    "expected_status": (example.get("expected_status")),
                    "reference_answer": (example.get("reference_answer")),
                },
            }
        )

    client.create_examples(
        dataset_id=dataset.id,
        examples=langsmith_examples,
    )

    print(f"Created dataset: {DATASET_NAME}")

    print(f"Created examples: {len(langsmith_examples)}")


def target(
    inputs: dict,
) -> dict:
    result = graph.invoke(
        {
            "query": inputs["query"],
            "retry_count": 0,
        }
    )

    documents = result.get(
        "documents",
        [],
    )

    retrieved_context = "\n\n---\n\n".join(
        document.page_content for document in documents
    )

    return {
        "answer": result.get("answer"),
        "route": result.get("route"),
        "status": result.get("status"),
        "retry_count": result.get(
            "retry_count",
            0,
        ),
        "evidence_sufficient": result.get("evidence_sufficient"),
        "evidence_reason": result.get("evidence_reason"),
        "retrieved_context": (retrieved_context),
    }


def route_correct(
    outputs: dict,
    reference_outputs: dict,
) -> bool:
    actual_route = outputs.get("route")

    expected_route = reference_outputs.get("expected_route")

    return actual_route == expected_route


def status_correct(
    outputs: dict,
    reference_outputs: dict,
) -> bool:
    expected_status = reference_outputs.get("expected_status")

    if expected_status is None:
        return True

    actual_status = outputs.get("status")

    return actual_status == expected_status


def retry_efficiency(
    outputs: dict,
    reference_outputs: dict,
) -> bool:
    expected_route = reference_outputs.get("expected_route")

    expected_status = reference_outputs.get("expected_status")

    retry_count = outputs.get(
        "retry_count",
        0,
    )

    # General requests should never
    # enter the RAG retry loop.
    if expected_route == "general":
        return retry_count == 0

    # Queries known to be answerable from
    # the knowledge base should ideally
    # succeed on the first retrieval.
    if expected_status == "success":
        return retry_count == 0

    # Queries whose answer is not present
    # in the KB may use the configured
    # retries before giving up safely.
    if expected_status == "insufficient_evidence":
        return retry_count <= 2

    return True


def answer_correctness(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    print("\n--- CORRECTNESS DEBUG ---")
    print("INPUTS:", inputs)
    print("OUTPUTS:", outputs)
    print("REFERENCE:", reference_outputs)

    reference_answer = reference_outputs.get("reference_answer")

    generated_answer = outputs.get("answer")

    print("GENERATED ANSWER:", generated_answer)
    reference_answer = reference_outputs.get("reference_answer")

    # General-agent queries are currently
    # outside this RAG answer benchmark.
    if reference_answer is None:
        return {
            "key": "answer_correctness",
            "score": 1.0,
            "comment": ("No reference answer was defined for this example."),
        }

    result = correctness_judge.invoke(
        f"""
Evaluate the generated answer against the
reference answer.

User question:
{inputs["query"]}

Reference answer:
{reference_answer}

Generated answer:
{outputs.get("answer", "")}

Evaluate semantic correctness rather than
exact wording.

Consider whether the generated answer:
- correctly answers the user's question,
- preserves the important facts in the
  reference answer,
- avoids contradicting the reference answer.

Scoring:
1.0 = fully correct
0.5 = partially correct
0.0 = incorrect or contradictory
"""
    )

    return {
        "key": "answer_correctness",
        "score": result.score,
        "comment": result.reason,
    }


def answer_groundedness(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    expected_route = reference_outputs.get("expected_route")

    # Groundedness against retrieved context
    # only applies to the RAG route.
    if expected_route != "rag":
        return {
            "key": "answer_groundedness",
            "score": 1.0,
            "comment": ("Groundedness evaluation is not required for general queries."),
        }

    context = outputs.get(
        "retrieved_context",
        "",
    )

    answer = outputs.get(
        "answer",
        "",
    )

    result = groundedness_judge.invoke(
        f"""
Evaluate whether the generated answer is
grounded in the retrieved evidence.

Retrieved evidence:
{context}

Generated answer:
{answer}

Judge ONLY against the retrieved evidence.

Do not use external knowledge.

Every factual claim in the answer should be
supported by the retrieved evidence.

It is acceptable for the answer to state that
the available evidence is insufficient when
the requested information is not present.

Scoring:
1.0 = every material factual claim is supported
0.5 = partly supported, with some unsupported claims
0.0 = major claims are unsupported or contradicted
"""
    )

    return {
        "key": "answer_groundedness",
        "score": result.score,
        "comment": result.reason,
    }


def main() -> None:
    client = Client()

    create_dataset_if_needed(client)

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            route_correct,
            status_correct,
            retry_efficiency,
            answer_correctness,
            answer_groundedness,
        ],
        experiment_prefix=("agentic-rag-v2"),
        metadata={
            "architecture": ("supervisor-general-rag-subgraph"),
            "max_retries": 2,
            "retrieval": ("fine-tuned-dense-bm25-rrf"),
            "generation_model": ("gpt-4.1-mini"),
            "judge_model": ("gpt-4.1-mini"),
        },
    )

    print("\nEvaluation completed.")

    print(results)


if __name__ == "__main__":
    main()
