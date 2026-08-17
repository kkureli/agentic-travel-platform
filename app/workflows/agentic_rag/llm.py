from langchain_openai import ChatOpenAI

from app.core.config import get_settings

DEFAULT_MODEL = "gpt-4.1-mini"


def get_chat_model() -> ChatOpenAI:
    settings = get_settings()

    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Set it in .env or the environment."
        )

    return ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=0,
        api_key=settings.openai_api_key,
    )
