from typing import Literal

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.rag.embeddings.embedding_service import (
    EmbeddingService,
)
from app.rag.retrieval.qdrant_hybrid_retriever import (
    QdrantHybridRetriever,
)
from app.rag.vector_store.qdrant_store import (
    QdrantStore,
)
from app.workflows.agentic_rag.general_agent import (
    general_agent,
)
from app.workflows.agentic_rag.rag_agent import (
    build_rag_agent,
)
from app.workflows.agentic_rag.state import (
    AgenticRAGState,
)
from app.workflows.agentic_rag.supervisor import (
    supervisor_node,
)

EMBEDDING_MODEL = "models/travel-embedding-triplet-v1/final"


def route_to_agent(
    state: AgenticRAGState,
) -> Literal[
    "general_agent",
    "rag_agent",
]:
    if state["route"] == "rag":
        return "rag_agent"

    return "general_agent"


def build_agentic_rag_graph():
    store = QdrantStore()

    embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL)

    retriever = QdrantHybridRetriever(
        store=store,
        dense_embedding_service=(embedding_service),
    )

    rag_agent = build_rag_agent(retriever)

    graph = StateGraph(AgenticRAGState)

    graph.add_node(
        "supervisor",
        supervisor_node,
    )

    graph.add_node(
        "general_agent",
        general_agent,
    )

    # compiled RAG subgraph
    graph.add_node(
        "rag_agent",
        rag_agent,
    )

    graph.add_edge(
        START,
        "supervisor",
    )

    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "general_agent": "general_agent",
            "rag_agent": "rag_agent",
        },
    )

    graph.add_edge(
        "general_agent",
        END,
    )

    graph.add_edge(
        "rag_agent",
        END,
    )

    return graph.compile()
