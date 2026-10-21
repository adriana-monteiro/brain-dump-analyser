
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

class Settings(BaseSettings):

    # SQLite
    database_path: str = str(BASE_DIR / "braindump.db")

    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"

    @field_validator("openai_api_key")
    @classmethod
    def _require_nonempty_key(cls, value: str) -> str:
        # Treat a blank key the same as a missing one, so copying
        # .env.example (which ships an empty OPENAI_API_KEY=) without filling
        # it in fails fast with a clear message instead of a later 401.
        if not value or not value.strip():
            raise ValueError("OPENAI_API_KEY must not be empty")
        return value

    # Retrieval / ChromaDB
    chroma_persist_directory: str = str(BASE_DIR / ".chroma" / "tasks")
    chroma_collection_name: str = "brain_dump_tasks"
    retrieval_top_k: int = 4
    retrieval_min_similarity: float = 0.3

    # Brain-dump defaults
    default_topics: list[str] = [
        "Work",
        "Personal",
        "Health",
        "Finance",
    ]

    # Append the interpret agent's step trace to chat replies. On by default;
    # set SHOW_INTERPRET_TRACE=0/false to disable.
    show_interpret_trace: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        # The only required setting is the API key. Turn pydantic's raw
        # "field required" traceback into an actionable message so a first-time
        # user (or any entry point: API, eval runner, scripts) sees what to do.
        key_problem = any(
            error.get("loc") == ("openai_api_key",)
            for error in exc.errors()
        )
        if key_problem:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
                "your OpenAI API key (or export OPENAI_API_KEY)."
            ) from exc
        raise


settings = get_settings()
