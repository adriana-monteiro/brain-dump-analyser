import logging
import os
import re

from backend import llm
from backend import repositories as repo
from backend.retrieval import RetrievalService

log = logging.getLogger("braindump")

BIG_WORDS = int(os.getenv("BIG_DUMP_WORDS", "10e10000"))
KINDS = {"note", "idea", "task"}
OPS = {"new", "update", "duplicate", "complete", "cancel", "needs_human"}

TEMPLATE = (
    'I turn brain dumps into organised notes, ideas and tasks. '
    'Send me whatever is on your mind '
    '(e.g. "call Sam Friday, buy milk, idea: weekly review") '
    "and I'll sort it for you."
)

ALL_YES = re.compile(
    r"^\s*(yes|y|ok|okay|confirm( all)?|accept( all)?|looks good|all good)\W*$",
    re.I,
)
ALL_NO = re.compile(
    r"^\s*(no|n|reject( all)?|discard( all)?|none|nothing)\W*$",
    re.I,
)


class BrainDumpService:
    def __init__(self, retrieval: RetrievalService | None = None):
        self.retrieval = retrieval or RetrievalService()

    # ---------- Public application API ----------

    def chat(self, conversation_id, message):
        conv = repo.get_or_create_conversation(conversation_id)
        reply = self.handle(conv, message)
        candidates = self.pending(conv)
        conversation = repo.get_conversation(conv)

        return (
            conv,
            reply,
            bool(conversation["closed"]),
            candidates,
        )

    def reset_conversation(self, conv_id):
        """Reset a conversation and return a fresh conversation ID."""
        if conv_id is not None:
            reset = repo.reset_conversation(conv_id)

            if not reset:
                raise LookupError(
                    f"Conversation {conv_id} not found"
                )

        # Create a fresh conversation with no inherited context.
        return repo.get_or_create_conversation()
        
    def resolve_candidate(self, candidate_id, action, text=None):
        return self.ui_resolve(candidate_id, action, text)

    def get_pending(self, conv_id):
        return self.pending(conv_id)

    def get_items(self, kind=None, status=None):
        return repo.get_items(kind=kind, status=status)

    def get_tasks(self):
        return repo.get_open_tasks()

    def get_topics(self):
        return repo.get_topics()

    def set_topics(self, topics):
        repo.set_topics(topics)

    def find_similar_tasks(self, text, topic=None, n_results=5):
        return self.retrieval.search_tasks(
            text=text,
            topic=topic,
            n_results=n_results,
        )

    # ---------- Persistence helpers ----------

    def say(self, conv, role, text):
        repo.add_message(conv, role, text)

    def pending(self, conv):
        return repo.get_pending_candidates(conv)

    def close(self, conv):
        repo.close_conversation(conv)

    # ---------- Main conversation workflow ----------

    def handle(self, conv, text):
        self.say(conv, "user", text)

        pend = self.pending(conv)
        lines = []
        other = text
        new = None

        if pend:
            other, lines = self.interpret_flow(conv, pend, text)

        if other:
            new = self.process_dump(conv, other)

            if new is None:
                lines.append(TEMPLATE)
            elif not new:
                lines.append("I couldn't find any items in that.")

        reply = self.compose(conv, lines, new)

        if pend and not self.pending(conv):
            self.close(conv)
            reply += (
                "\n\n✅ All items resolved — this chat is finished. "
                "Send a new message to start another."
            )

        self.say(conv, "assistant", reply)
        return reply

    # ---------- Extract brain dumps ----------

    def process_dump(self, conv, text):
        """Return None for a non-dump, otherwise return candidate IDs."""
        topics = self.get_topics() + ["INBOX"]

        if not llm.is_brain_dump(text):
            return None

        big = len(text.split()) > BIG_WORDS
        items = llm.extract(text, topics, classify=not big)

        if big:
            kinds = llm.classify(items)
            for item, kind in zip(items, kinds):
                item["kind"] = kind

        # Uses all candidates, including previously resolved candidates.
        number = repo.get_max_display_no(conv)
        candidate_ids = []

        for item in items:
            item_text = str(item.get("text", "")).strip()
            if not item_text:
                continue

            number += 1

            kind = item.get("kind")
            if kind not in KINDS:
                kind = "note"

            topic = item.get("topic")
            if topic not in topics:
                topic = "INBOX"

            candidate_id = repo.create_candidate(
                conv_id=conv,
                display_no=number,
                kind=kind,
                text=item_text,
                topic=topic,
            )
            candidate_ids.append(candidate_id)

            if kind == "task":
                self.judge(candidate_id)

        return candidate_ids

    # ---------- Task retrieval and judging ----------

    def judge(self, candidate_id):
        candidate = repo.get_candidate(candidate_id)
        if not candidate:
            return

        similar = self.find_similar_tasks(
            text=candidate["text"],
            topic=candidate["topic"],
            n_results=3,
        )

        # With no likely matches, keep the default "new" operation.
        if not similar:
            return

        similar_ids = {item["id"] for item in similar}

        try:
            result = llm.judge(candidate["text"], similar)

            operation = result.get("operation")
            task_id = result.get("task_id")

            if operation not in OPS:
                raise ValueError(f"Invalid task operation: {operation}")

            if operation in {"update", "duplicate", "complete", "cancel"}:
                if task_id not in similar_ids:
                    raise ValueError("Judged task is not in retrieval results")

            if operation == "needs_human" and not result.get("question"):
                raise ValueError("Missing clarification question")

            if operation == "new" or task_id not in similar_ids:
                task_id = None

            repo.update_candidate(
                    candidate_id,
                    op=operation,
                    item_id=task_id,
                    proposed_text=result.get("proposed_text"),
                    question=(
                        result.get("question")
                        if operation == "needs_human"
                        else None
                    ),
                )

        except Exception:
            # Keep the candidate available for the user to review.
            log.warning(
                "Task judging failed; leaving candidate as NEW",
                exc_info=True,
            )

    # ---------- Confirming and rejecting ----------

    def confirm(self, candidate):
        kind = candidate["kind"]
        if kind not in KINDS:
            raise ValueError(f"Invalid item kind: {kind}")
        
        operation = (
            "new"
            if candidate["op"] == "needs_human"
            else candidate["op"]
        )
        item_id = candidate.get("item_id")
        message = "saved"

        if candidate["kind"] == "task":
            if operation == "new":
                item_id = repo.create_task(
                    candidate["text"],
                    candidate["topic"],
                )
                message = "task created"

            elif operation == "update":
                if item_id is None:
                    raise ValueError("Missing item ID for task update")

                repo.update_task_text(
                    item_id,
                    candidate.get("proposed_text") or candidate["text"],
                )
                message = "task updated"

            elif operation == "complete":
                if item_id is None:
                    raise ValueError("Missing item ID for task completion")

                repo.set_task_status(item_id, "done")
                message = "task completed"

            elif operation == "cancel":
                if item_id is None:
                    raise ValueError("Missing item ID for task cancellation")

                repo.set_task_status(item_id, "cancelled")
                message = "task cancelled"

            else:
                message = "duplicate, nothing to do"

        elif operation == "duplicate":
            message = "duplicate, nothing to do"

        else:
            item_id = repo.create_item(
                kind=candidate["kind"],
                text=candidate["text"],
                topic=candidate["topic"],
            )
            message = f"{candidate['kind']} saved"

        repo.update_candidate(
            candidate["id"],
            status="confirmed",
            item_id=item_id,
            question=None,
        )

        return f"#{candidate['display_no']}: {message}"
    def reject(self, candidate):
        repo.update_candidate(
            candidate["id"],
            status="rejected",
            question=None,
        )
        return f"#{candidate['display_no']}: rejected"

    def ui_resolve(self, candidate_id, action, text=None):
        candidate = repo.get_candidate(candidate_id)

        if not candidate or candidate["status"] != "pending":
            raise KeyError("candidate not pending")

        if action == "confirm":
            if candidate.get("question"):
                raise ValueError(
                    "Answer the clarifying question in chat first"
                )
            self.confirm(candidate)

        elif action == "reject":
            self.reject(candidate)

        elif action == "update" and text and text.strip():
            repo.update_candidate(
                candidate_id,
                text=text.strip(),
            )

        else:
            raise ValueError("bad action")

        remaining = self.pending(candidate["conv_id"])
        closed = not remaining

        if closed:
            self.close(candidate["conv_id"])

        return candidate["conv_id"], closed

    # ---------- Interpret replies to pending candidates ----------

    def apply_actions(self, actions, candidates):
        by_number = {
            candidate["display_no"]: candidate
            for candidate in candidates
        }
        done = set()
        output = []

        for action in actions if isinstance(actions, list) else []:
            if not isinstance(action, dict):
                continue

            candidate = by_number.get(action.get("display_no"))
            operation = action.get("operation")
            text = (action.get("text") or "").strip()

            if not candidate or candidate["id"] in done:
                continue

            if operation == "confirm":
                output.append(self.confirm(candidate))

            elif operation == "reject":
                output.append(self.reject(candidate))

            elif operation == "update_candidate" and text:
                repo.update_candidate(
                    candidate["id"],
                    text=text,
                    question=None,
                )
                output.append(f"#{candidate['display_no']}: updated")
                continue

            elif (
                candidate["kind"] == "task"
                and operation in {
                    "update_task",
                    "complete_task",
                    "cancel_task",
                }
                and candidate.get("item_id")
            ):
                task_operation = operation.removesuffix("_task")

                modified = {
                    **candidate,
                    "kind": "task",
                    "op": task_operation,
                    "proposed_text": text or candidate["text"],
                }
                output.append(self.confirm(modified))

            elif operation == "needs_human" and action.get("question"):
                repo.update_candidate(
                    candidate["id"],
                    op="needs_human",
                    question=action["question"],
                )
                output.append(
                    f"#{candidate['display_no']}: needs your input"
                )
                continue

            else:
                # Invalid actions leave the candidate pending.
                continue

            done.add(candidate["id"])

        return output

    def interpret_flow(self, conv, candidates, text):
        """Return (other content, summary lines)."""
        free = [
            candidate
            for candidate in candidates
            if not candidate.get("question")
        ]

        mode = (
            "confirm" if ALL_YES.match(text)
            else "reject" if ALL_NO.match(text)
            else None
        )

        if mode and free:
            method = self.confirm if mode == "confirm" else self.reject
            return "", [method(candidate) for candidate in free]

        try:
            model_candidates = [
                {
                    key: candidate.get(key)
                    for key in (
                        "display_no",
                        "kind",
                        "text",
                        "op",
                        "proposed_text",
                        "question",
                        "existing_item_text",
                        "existing_item_kind",
                        "existing_item_status",
                    )
                }
                for candidate in candidates
            ]

            result = llm.interpret(
                model_candidates,
                repo.get_history(conv),
                text,
            )

            lines = self.apply_actions(
                result.get("actions"),
                candidates,
            )

            return (result.get("other_content") or "").strip(), lines

        except Exception:
            log.warning("Reply interpretation failed", exc_info=True)
            return "", [
                "Sorry, I couldn't interpret that. Try `confirm all`, "
                "`reject #2`, or use the buttons."
            ]

    # ---------- Format the assistant response ----------

    def describe(self, candidate):
        description = (
            f"**#{candidate['display_no']}** "
            f"[{candidate['kind']}] {candidate['text']} "
            f"_({candidate['topic']})_"
        )

        task_text = candidate.get("existing_item_text")
        operation = candidate["op"]

        if candidate["kind"] == "task" and task_text:
            descriptions = {
                "update": f" → looks like an update to existing task “{task_text}”",
                "duplicate": f" → looks like a duplicate of “{task_text}”",
                "complete": f" → looks like completing “{task_text}”",
                "cancel": f" → looks like cancelling “{task_text}”",
            }

            description += descriptions.get(operation, "")

            if operation == "update" and candidate.get("proposed_text"):
                description += (
                    f" (proposed: “{candidate['proposed_text']}”)"
                )

        if candidate.get("question"):
            description += f"\n  ❓ {candidate['question']}"

        return description

    def compose(self, conv, lines, new):
        candidates = self.pending(conv)
        parts = []

        if lines:
            parts.append("\n".join(f"- {line}" for line in lines))

        if candidates:
            heading = (
                "I have gathered these items from your brain dump:"
                if new
                else "Still open:"
            )
            parts.append(
                heading
                + "\n\n"
                + "\n".join(
                    f"- {self.describe(candidate)}"
                    for candidate in candidates
                )
            )
            parts.append(
                "_Reply e.g. `confirm all`, `reject #2`, or answer "
                "the questions — or use the panel._"
            )

        return "\n\n".join(parts)

    
    def update_task_status(self, task_id: int, status: str):
        if status not in {"open", "done", "cancelled"}:
            raise ValueError("Status must be open, done, or cancelled")

        # Verify that the task exists before updating it.
        task = repo.get_task(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")

        repo.set_task_status(task_id, status)
        return {"id": task_id, "status": status}
