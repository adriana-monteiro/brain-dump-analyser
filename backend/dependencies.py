
from functools import lru_cache

from backend.services import BrainDumpService


@lru_cache
def get_brain_dump_service() -> BrainDumpService:
    return BrainDumpService()
