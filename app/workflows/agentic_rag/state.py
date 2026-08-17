from typing import Literal, TypedDict

from langchain_core.documents import Document


class AgenticRAGState(TypedDict, total=False):
    query: str

    route: Literal["general", "rag"]

    documents: list[Document]
    rewritten_query: str

    evidence_sufficient: bool
    evidence_reason: str

    retry_count: int

    status: Literal[
        "success",
        "insufficient_evidence",
    ]

    answer: str
