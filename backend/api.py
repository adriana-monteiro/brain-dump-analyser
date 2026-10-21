
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_brain_dump_service
from backend.services import BrainDumpService
from backend.schemas.api import ChatIn, ChatOut, ResolveIn, TaskOut, TaskStatusIn, ItemOut, CandidateKindIn, CandidateTopicIn, LLMCallOut, MetricsSummaryOut

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
    try:
        service.set_topics(topics)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return service.get_topics()


@router.get("/tasks", response_model=list[TaskOut])
def get_tasks(service: BrainDump):
    return service.get_tasks()


@router.post("/chat", response_model=ChatOut)
def chat(b: ChatIn, service: BrainDump):
    conv, reply, closed, candidates, trace = service.chat(
        b.conversation_id,
        b.message,
    )

    return {
        "conversation_id": conv,
        "reply": reply,
        "candidates": candidates,
        "closed": closed,
        "trace": trace,
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


@router.get("/conversations/latest-open")
def get_latest_open_conversation(service: BrainDump):
    """Latest open conversation with pending items (hydrated), or
    `{"conversation_id": None}` when there is nothing to resume."""
    conversation = service.get_latest_open_conversation()
    if conversation is None:
        return {"conversation_id": None}

    conv_id = conversation["id"]
    return {
        "conversation_id": conv_id,
        "closed": bool(conversation["closed"]),
        "messages": service.get_messages(conv_id),
        "candidates": service.get_pending(conv_id),
    }


@router.get("/conversations/{conv_id}/messages")
def get_conversation(conv_id: int, service: BrainDump):
    """Conversation state for rehydration: transcript (oldest first), pending
    candidates, closed flag. 404 if it doesn't exist."""
    conversation = service.get_conversation(conv_id)
    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conv_id} not found",
        )

    return {
        "conversation_id": conv_id,
        "closed": bool(conversation["closed"]),
        "messages": service.get_messages(conv_id),
        "candidates": service.get_pending(conv_id),
    }


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


@router.patch("/candidates/{cid}/kind", response_model=ChatOut)
def change_candidate_kind(cid: int, body: CandidateKindIn, service: BrainDump):
    try:
        conv_id = service.change_candidate_kind(cid, body.kind)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "conversation_id": conv_id,
        "reply": None,
        "candidates": service.get_pending(conv_id),
        "closed": False,
    }


@router.post("/candidates/{cid}/force-new", response_model=ChatOut)
def force_new_candidate(cid: int, service: BrainDump):
    """Save a candidate as a new item, overriding the judge's match
    (the panel's 'Force create new task' control)."""
    try:
        conv_id, closed = service.force_new_candidate(cid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    reply = (
        "✅ All items resolved — chat finished." if closed else None
    )
    return {
        "conversation_id": conv_id,
        "reply": reply,
        "candidates": service.get_pending(conv_id),
        "closed": closed,
    }


@router.patch("/candidates/{cid}/topic", response_model=ChatOut)
def change_candidate_topic(cid: int, body: CandidateTopicIn, service: BrainDump):
    try:
        conv_id, applied = service.change_candidate_topic(cid, body.topic)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Tell the user if their requested topic was too long and fell back.
    requested = body.topic.strip().title()
    reply = (
        None
        if applied == requested or requested == "INBOX"
        else f"'{body.topic}' couldn't be used as a topic — filed under {applied}."
    )

    return {
        "conversation_id": conv_id,
        "reply": reply,
        "candidates": service.get_pending(conv_id),
        "closed": False,
    }


@router.get("/metrics", response_model=list[LLMCallOut])
def list_metrics(
    service: BrainDump,
    step: str | None = None,
    model: str | None = None,
    status: str | None = None,
    limit: int = 100,
):
    if status is not None and status not in {"ok", "error"}:
        raise HTTPException(
            status_code=400,
            detail="status must be 'ok' or 'error'",
        )

    return service.list_metrics(step=step, model=model, status=status, limit=limit)


@router.get("/metrics/summary", response_model=MetricsSummaryOut)
def metrics_summary(
    service: BrainDump,
    step: str | None = None,
    model: str | None = None,
):
    return service.metrics_summary(step=step, model=model)
