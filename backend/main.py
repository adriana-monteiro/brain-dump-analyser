
from contextlib import asynccontextmanager

from fastapi import FastAPI
from backend.database import init_db
from backend.api import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Brain dump MVP",
    lifespan=lifespan,
)

app.include_router(router)
