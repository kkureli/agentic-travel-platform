from typing import Literal

from pydantic import BaseModel, Field

from app.workflows.agentic_rag.llm import get_chat_model
from app.workflows.agentic_rag.state import (
    AgenticRAGState,
)


class RouteDecision(BaseModel):
    route: Literal[
        "general",
        "rag",
    ] = Field(
        description=(
            "Use 'rag' for questions requiring "
            "travel-platform policies, booking rules, "
            "refund rules, insurance rules, loyalty rules, "
            "or other private platform knowledge. "
            "Use 'general' for casual conversation "
            "or general knowledge."
        )
    )


llm = get_chat_model()

router_llm = llm.with_structured_output(RouteDecision)


def supervisor_node(
    state: AgenticRAGState,
) -> dict:
    decision = router_llm.invoke(
        f"""
Classify the user's request.

User query:
{state["query"]}

Routing policy:

Use "rag" whenever the query asks about,
implies, or depends on travel-platform-specific
information, including:

- hotels
- reservations
- booking rules
- cancellation
- refunds
- compensation
- check-in / check-out
- payments
- insurance
- loyalty
- customer support
- benefits
- hotel services
- platform policies

IMPORTANT:
If the user asks whether a hotel/platform provides
a benefit, compensation, service, rule, or policy,
route to "rag" even if the claim sounds unusual,
unlikely, or unsupported.

The RAG agent is responsible for determining
whether the knowledge base contains sufficient
evidence.

Use "general" only for:
- greetings
- casual conversation
- general knowledge that does not depend on
  platform-specific information.

When uncertain, prefer "rag".
"""
    )

    return {
        "route": decision.route,
        "retry_count": 0,
    }
