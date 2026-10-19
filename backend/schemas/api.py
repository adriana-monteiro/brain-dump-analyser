
"""HTTP API schemas: the contract between the Streamlit UI and FastAPI."""

from typing import Literal

from pydantic import BaseModel

from .llm import JudgeOp, Kind


# ---- shared types ----

ItemStatus = Literal["open", "done", "cancelled"]


# ---- requests ----

class ChatIn(BaseModel):
    message: str
    conversation_id: int | None = None


class ResolveIn(BaseModel):
    action: Literal["confirm", "reject", "update"]
    text: str | None = None


class TaskStatusIn(BaseModel):
    status: ItemStatus


# ---- responses ----

class CandidateOut(BaseModel):
    id: int
    conv_id: int
    display_no: int
    kind: Kind
    text: str
    topic: str
    status: Literal["pending", "confirmed", "rejected"]
    op: JudgeOp
    item_id: int | None = None
    existing_item_text: str | None = None
    existing_item_kind: Kind | None = None
    existing_item_status: ItemStatus | None = None
    proposed_text: str | None = None
    question: str | None = None


class ItemOut(BaseModel):
    id: int
    kind: Kind
    text: str
    topic: str
    status: ItemStatus | None = None
    created_at: str


class TaskOut(BaseModel):
    id: int
    text: str
    topic: str
    status: ItemStatus


class ChatOut(BaseModel):
    conversation_id: int
    reply: str | None = None
    candidates: list[CandidateOut]
    closed: bool
