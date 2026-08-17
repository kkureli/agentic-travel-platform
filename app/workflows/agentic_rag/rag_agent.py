from typing import Literal

from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from pydantic import BaseModel, Field

from app.rag.retrieval.qdrant_hybrid_retriever import (
    QdrantHybridRetriever,
)
from app.workflows.agentic_rag.llm import get_chat_model
from app.workflows.agentic_rag.state import (
    AgenticRAGState,
)


class EvidenceDecision(BaseModel):
    sufficient: bool = Field(
        description=(
            "Whether the retrieved evidence is sufficient "
            "to answer the user's question accurately."
        )
    )

    reason: str = Field(
        description=(
            "A concise explanation of why the evidence is or is not sufficient."
        )
    )


class RewrittenQuery(BaseModel):
    query: str = Field(
        description=(
            "A rewritten search query that preserves "
            "the original intent but improves retrieval."
        )
    )


llm = get_chat_model()


evidence_llm = llm.with_structured_output(EvidenceDecision)

rewrite_llm = llm.with_structured_output(RewrittenQuery)


def build_rag_agent(
    retriever: QdrantHybridRetriever,
):
    def retrieve_node(
        state: AgenticRAGState,
    ) -> dict:
        query = state.get("rewritten_query") or state["query"]

        results = retriever.search(
            query=query,
            tenant_id="travel-platform",
            top_k=5,
        )

        documents = [result.document for result in results]

        return {
            "documents": documents,
        }

    def grade_evidence_node(
        state: AgenticRAGState,
    ) -> dict:
        context = "\n\n---\n\n".join(
            document.page_content
            for document in state.get(
                "documents",
                [],
            )
        )

        decision = evidence_llm.invoke(
            f"""
You are evaluating retrieved evidence for a
travel-platform knowledge base.

User question:
{state["query"]}

Retrieved evidence:
{context}

Determine whether the evidence contains enough
information to answer the user's actual question.

Rules:
- Do not mark evidence sufficient merely because
  it is topically related.
- The evidence must support the specific answer.
- If the requested policy or fact is absent,
  return sufficient=false.
"""
        )

        return {
            "evidence_sufficient": decision.sufficient,
            "evidence_reason": decision.reason,
        }

    def rewrite_query_node(
        state: AgenticRAGState,
    ) -> dict:
        rewritten = rewrite_llm.invoke(
            f"""
Rewrite this query to improve retrieval from
a travel-platform policy knowledge base.

Preserve the user's original intent.
Do not answer the question.
Do not add unsupported facts.

Original query:
{state["query"]}
"""
        )

        return {
            "rewritten_query": rewritten.query,
            "retry_count": (
                state.get(
                    "retry_count",
                    0,
                )
                + 1
            ),
        }

    def generate_answer_node(
        state: AgenticRAGState,
    ) -> dict:
        context = "\n\n---\n\n".join(
            document.page_content
            for document in state.get(
                "documents",
                [],
            )
        )

        if state.get("status") == "insufficient_evidence":
            response = llm.invoke(
                [
                    (
                        "system",
                        (
                            "The knowledge base does not contain "
                            "sufficient evidence to answer the "
                            "user's question. Clearly state that "
                            "the available information is "
                            "insufficient. Do not invent a policy."
                        ),
                    ),
                    (
                        "human",
                        state["query"],
                    ),
                ]
            )

            return {
                "answer": response.content,
            }

        response = llm.invoke(
            [
                (
                    "system",
                    (
                        "Answer using only the retrieved "
                        "travel-platform evidence. "
                        "Do not add unsupported policy claims."
                    ),
                ),
                (
                    "human",
                    f"""
Question:
{state["query"]}

Evidence:
{context}
""",
                ),
            ]
        )

        return {
            "status": "success",
            "answer": response.content,
        }

    def route_after_grade(
        state: AgenticRAGState,
    ) -> Literal[
        "generate",
        "rewrite",
        "insufficient",
    ]:
        if state.get(
            "evidence_sufficient",
            False,
        ):
            return "generate"

        if (
            state.get(
                "retry_count",
                0,
            )
            >= 2
        ):
            return "insufficient"

        return "rewrite"

    def mark_insufficient_evidence_node(
        state: AgenticRAGState,
    ) -> dict:
        return {
            "status": "insufficient_evidence",
        }

    graph = StateGraph(AgenticRAGState)

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "grade",
        grade_evidence_node,
    )

    graph.add_node(
        "rewrite",
        rewrite_query_node,
    )

    graph.add_node(
        "generate",
        generate_answer_node,
    )

    graph.add_node(
        "insufficient",
        mark_insufficient_evidence_node,
    )

    graph.add_edge(
        START,
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        "grade",
    )

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "insufficient": "insufficient",
        },
    )

    graph.add_edge(
        "rewrite",
        "retrieve",
    )

    graph.add_edge(
        "insufficient",
        "generate",
    )

    graph.add_edge(
        "generate",
        END,
    )

    return graph.compile()
