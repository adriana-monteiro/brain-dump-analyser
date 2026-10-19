
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_brain_dump_service
from backend.services import BrainDumpService
from backend.schemas.api import ChatIn, ChatOut, ResolveIn, TaskOut, TaskStatusIn, ItemOut

router = APIRouter()

BrainDump = Annotated[
    BrainDumpService,
    Depends(get_brain_dump_service),
]


@router.get("/settings/topics", response_model=list[str])
def get_topics(service: BrainDump):
    return service.get_topics()


@router.put("/settings/topics", response_model=list[str])
def set_topics(topics: list[str], service: BrainDump):
    service.set_topics(topics)
    return service.get_topics()


@router.get("/tasks", response_model=list[TaskOut])
def get_tasks(service: BrainDump):
    return service.get_tasks()


@router.post("/chat", response_model=ChatOut)
def chat(b: ChatIn, service: BrainDump):
    conv, reply, closed, candidates = service.chat(
        b.conversation_id,
        b.message,
    )

    return {
        "conversation_id": conv,
        "reply": reply,
        "candidates": candidates,
        "closed": closed,
    }


@router.post("/candidates/{cid}/resolve", response_model=ChatOut)
def resolve(cid: int, b: ResolveIn, service: BrainDump):
    try:
        conv, closed = service.resolve_candidate(
            cid,
            b.action,
            b.text,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    reply = (
        "✅ All items resolved — chat finished."
        if closed
        else None
    )

    return {
        "conversation_id": conv,
        "reply": reply,
        "candidates": service.get_pending(conv),
        "closed": closed,
    }


@router.get("/conversations/{conv_id}/candidates")
def get_candidates(conv_id: int, service: BrainDump):
    try:
        return service.get_pending(conv_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conv_id} not found",
        ) from exc


@router.patch("/tasks/{task_id}/status")
def update_task_status(
    task_id: int,
    body: TaskStatusIn,
    service: BrainDump,
):
    try:
        return service.update_task_status(task_id, body.status)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/items", response_model=list[ItemOut])
def get_items(
    service: BrainDump,
    kind: str | None = None,
    status: str | None = None,
):
    valid_kinds = {"task", "note", "idea"}
    valid_statuses = {"open", "done", "cancelled"}

    if kind is not None and kind not in valid_kinds:
        raise HTTPException(
            status_code=400,
            detail="Kind must be task, note, or idea",
        )

    if status is not None and status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail="Status must be open, done, or cancelled",
        )

    return service.get_items(kind=kind, status=status)

@router.post("/conversations/{conv_id}/reset", response_model=ChatOut)
def reset_conversation(conv_id: int, service: BrainDump):
    try:
        new_conv_id = service.reset_conversation(conv_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return {
        "conversation_id": new_conv_id,
        "reply": "Conversation reset. You can start fresh.",
        "candidates": [],
        "closed": False,
    }