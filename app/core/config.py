# .env değerleri tutar, Settings bunları typed şekilde yükler,
# uygulamanın geri kalanı da merkezi settings objesinden config okur.
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic Travel Platform"
    environment: str = "development"

    database_url: str
    qdrant_url: str
    redis_url: str

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str

    openai_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
