
# backend/config.py

from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

class Settings(BaseSettings):
    # Application
    app_name: str = "Brain Dump MVP"
    debug: bool = False

    # SQLite
    database_path: str = str(BASE_DIR / "braindump.db")

    # OpenAI
    openai_api_key: str
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Retrieval / ChromaDB
    chroma_persist_directory: str = str(BASE_DIR / ".chroma" / "tasks")
    chroma_collection_name: str = "brain_dump_tasks"
    retrieval_top_k: int = 5
    retrieval_min_similarity: float = 0.35

    # Brain-dump defaults
    default_topics: list[str] = [
        "Work",
        "Personal",
        "Health",
        "Finance",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
