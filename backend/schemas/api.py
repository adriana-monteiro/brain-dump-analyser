
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


class CandidateKindIn(BaseModel):
    kind: Kind


class CandidateTopicIn(BaseModel):
    topic: str


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


class InterpretDecision(BaseModel):
    """One action the interpret agent took on a pending candidate."""
    display_no: int
    operation: str
    before_text: str | None = None
    after_text: str | None = None
    kind: str | None = None
    topic: str | None = None
    question: str | None = None


class InterpretTrace(BaseModel):
    """Structured, UI-facing record of how a chat turn was interpreted."""
    source: str
    decisions: list[InterpretDecision] = []
    new_dump: str = ""
    tools: list[str] = []
    steps: list[dict] = []


class ChatOut(BaseModel):
    conversation_id: int
    reply: str | None = None
    candidates: list[CandidateOut]
    closed: bool
    trace: InterpretTrace | None = None


class LLMCallOut(BaseModel):
    id: int
    step: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    latency_ms: int
    # Null when the model has no price entry; preserved rather than zeroed.
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    cost_usd: float | None = None
    status: Literal["ok", "error"]
    # Stored as 0/1 in SQLite; pydantic coerces to bool here.
    error: bool
    created_at: str


class MetricsBreakdownOut(BaseModel):
    """One per-step or per-model aggregate group; `key` is the step or model name."""
    key: str
    call_count: int
    cost_usd: float
    tokens: int


class MetricsSummaryOut(BaseModel):
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    avg_latency_ms: float
    call_count: int
    error_count: int
    by_step: list[MetricsBreakdownOut]
    by_model: list[MetricsBreakdownOut]
