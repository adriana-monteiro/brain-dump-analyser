
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from backend.database import init_db
from backend.api import router


def _configure_logging():
    """Surface the `braindump` logger on the console.

    LOG_LEVEL (default INFO) tunes verbosity: WARNING to quiet turn-by-turn
    decisions, DEBUG for the full LLM payloads in llm.py.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("braindump").setLevel(level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    init_db()
    yield


app = FastAPI(
    title="Brain dump MVP",
    lifespan=lifespan,
)

app.include_router(router)
