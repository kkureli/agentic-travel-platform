from app.rag.retrieval.query_rewriter import (
    QueryRewriter,
)
from app.services.llm_service import (
    LLMService,
)


def main() -> None:
    llm_service = LLMService()

    rewriter = QueryRewriter(
        llm_service=llm_service,
    )

    queries = [
        "Can I get my money back?",
        "What if hotel cancels?",
        "What do I need when booking?",
    ]

    for query in queries:
        rewritten_query = rewriter.rewrite(query)

        print("\nORIGINAL:")
        print(query)

        print("\nREWRITTEN:")
        print(rewritten_query)


if __name__ == "__main__":
    main()
