from pydantic import BaseModel


class Source(BaseModel):
    source: str
    score: float
    text: str


class RAGResponse(BaseModel):
    answer: str
    sources: list[Source]
