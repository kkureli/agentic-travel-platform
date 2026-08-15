from langchain_openai import ChatOpenAI

from app.core.config import get_settings


class LLMService:
    def __init__(self):
        settings = get_settings()
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Set it in .env or the environment."
            )

        self.model = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0,
            api_key=settings.openai_api_key,
        )

    def generate(self, messages):
        return self.model.invoke(messages)
