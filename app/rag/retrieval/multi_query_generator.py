from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class QueryVariants(BaseModel):
    queries: list[str] = Field(
        description="Alternative search queries preserving the original intent."
    )


class MultiQueryGenerator:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        query_count: int = 3,
    ):
        self.query_count = query_count

        llm = ChatOpenAI(
            model=model,
            temperature=0,
        )

        self.structured_llm = llm.with_structured_output(QueryVariants)

    def generate(
        self,
        query: str,
        query_count: int | None = None,
    ) -> list[str]:
        count = query_count or self.query_count

        prompt = f"""
Generate exactly {count} alternative search queries
for the following user query.

Requirements:
- Preserve the original user intent.
- Do not answer the question.
- Make each query useful for document retrieval.
- Use different wording or perspectives.
- Do not add facts that are not present in the original query.

Original query:
{query}
"""

        result = self.structured_llm.invoke(prompt)

        unique_queries: list[str] = []

        for generated_query in result.queries:
            generated_query = generated_query.strip()

            if not generated_query:
                continue

            if generated_query == query:
                continue

            if generated_query in unique_queries:
                continue

            unique_queries.append(generated_query)

            if len(unique_queries) >= count:
                break

        return unique_queries
