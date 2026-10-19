"""LLM output schemas: what the model must return per pipeline step (OpenAI strict mode: all fields required)."""
from typing import Literal
from pydantic import BaseModel

Kind = Literal["note", "idea", "task"]
JudgeOp = Literal["new", "update", "duplicate", "complete", "cancel", "needs_human"]
InterpretOp = Literal["confirm", "reject", "update_candidate", "update_task",
                      "complete_task", "cancel_task", "needs_human"]

Confidence = Literal["high", "low", "medium"]

class DumpOut(BaseModel):
    is_brain_dump: bool

class Item(BaseModel):
    text: str
    topic: str
    alternative_topic: str | None = None

class ItemWithKind(Item):
    kind: Kind
    confidence: Confidence
    reason: str | None = None

class ExtractOut(BaseModel):
    items: list[Item]

class ExtractClassifyOut(BaseModel):
    items: list[ItemWithKind]

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

class Action(BaseModel):
    display_no: int
    operation: InterpretOp
    text: str | None
    question: str | None

class InterpretOut(BaseModel):
    actions: list[Action]
    questions_answered: bool
    other_content: str

SCHEMAS = {"dump": DumpOut, "extract": ExtractOut, "extract_classify": ExtractClassifyOut,
           "classify": ClassifyOut, "judge": JudgeOut, "interpret": InterpretOut}