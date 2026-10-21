"""LLM output schemas: what the model must return per pipeline step (OpenAI strict mode: all fields required)."""
from typing import Literal
from pydantic import BaseModel

Kind = Literal["note", "idea", "task"]
JudgeOp = Literal["new", "update", "duplicate", "complete", "cancel", "needs_human"]
PlanOp = Literal["confirm", "reject", "change_text", "change_kind", "change_topic",
                 "needs_human"]


class DumpOut(BaseModel):
    is_brain_dump: bool

class Item(BaseModel):
    text: str
    topic: str
    alternative_topic: str | None = None

class ExtractOut(BaseModel):
    items: list[Item]

class ItemClassifyOut(BaseModel):
    text: str
    kind: Kind

class ClassifyOut(BaseModel):
    items: list[ItemClassifyOut]

class JudgeOut(BaseModel):
    operation: JudgeOp
    task_id: str | None
    proposed_text: str | None
    clarifying_question: str | None

class PlanAction(BaseModel):
    display_no: int
    operation: PlanOp
    text: str | None
    kind: Kind | None
    topic: str | None
    question: str | None

class SubmitPlan(BaseModel):
    actions: list[PlanAction]
    new_dump: str | None
    unrelated: str | None

class GetHistoryArgs(BaseModel):
    n: int

SCHEMAS = {"dump": DumpOut, "extract": ExtractOut,
           "classify": ClassifyOut, "judge": JudgeOut}