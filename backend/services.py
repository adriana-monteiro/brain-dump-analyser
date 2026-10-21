import logging

from backend import fast_path
from backend import llm
from backend import metrics
from backend import repositories as repo
from backend.config import settings
from backend.interpret_agent import InterpretAgent
from backend.retrieval import RetrievalService

log = logging.getLogger("braindump")

KINDS = {"note", "idea", "task"}
OPS = {"new", "update", "duplicate", "complete", "cancel", "needs_human"}
# Topics are short labels. A proposed topic longer than this falls back to INBOX.
TOPIC_MAX_WORDS = 3

SHOW_INTERPRET_TRACE = settings.show_interpret_trace


def _format_trace(result) -> str:
    """One compact, user-visible line per step of the agent's trace.

    Each entry is `step·tool·status` with the error reason or key args when
    present, e.g. `1·get_history·ok (n=3)` or `2·submit_plan·error: ...`.
    """
    parts = []
    for entry in result.trace:
        line = f"{entry['step']}/{entry['tool']}/{entry['status']}"
        detail = entry.get("detail")
        args = entry.get("args")
        if detail:
            line += f": {detail}"
        elif args:
            line += " (" + ", ".join(f"{k}={v}" for k, v in args.items()) + ")"
        parts.append(line)
    tools = ", ".join(result.tools_used) or "none"
    return "trace: " + " -> ".join(parts) + f"  [tools: {tools}]"


def _with_trace(lines: list[str], result) -> list[str]:
    """Append the debug trace line to the reply when SHOW_INTERPRET_TRACE is on."""
    if SHOW_INTERPRET_TRACE:
        return [*lines, _format_trace(result)]
    return lines


def _build_trace(result, candidates, source: str) -> dict:
    """UI-facing record of how a chat turn was interpreted.

    `source` is "fast_path", "agent", or "none". `decisions` pairs each action
    with its candidate, keeping `before_text` so a rewrite shows old and new.
    """
    by_no = {c["display_no"]: c for c in candidates}
    decisions = []
    new_dump = ""
    tools = []
    steps = []

    if result is not None:
        # Guard the coercion so a test double never turns trace building
        # (pure observability) into a turn-breaking error.
        try:
            tools = list(result.tools_used)
        except TypeError:
            tools = []
        try:
            steps = [s for s in result.trace if isinstance(s, dict)]
        except TypeError:
            steps = []
        plan = result.plan
        if plan is not None:
            new_dump = (plan.new_dump or "").strip()
            for action in plan.actions:
                candidate = by_no.get(action.display_no)
                decisions.append(
                    {
                        "display_no": action.display_no,
                        "operation": action.operation,
                        "before_text": (
                            candidate["text"] if candidate else None
                        ),
                        "after_text": (action.text or "").strip() or None,
                        "kind": getattr(action, "kind", None),
                        "topic": getattr(action, "topic", None),
                        "question": getattr(action, "question", None),
                    }
                )

    return {
        "source": source,
        "decisions": decisions,
        "new_dump": new_dump,
        "tools": tools,
        "steps": steps,
    }

TEMPLATE = (
    'I turn brain dumps into organised notes, ideas and tasks. '
    'Send me whatever is on your mind '
    '(e.g. "call Sam Friday, buy milk, idea: weekly review") '
    "and I'll sort it for you."
)

class BrainDumpService:
    def __init__(self, retrieval: RetrievalService | None = None):
        self.retrieval = retrieval or RetrievalService()
        self.agent = InterpretAgent()

    # ---------- Public application API ----------

    def chat(self, conversation_id, message):
        conv = repo.get_or_create_conversation(conversation_id)
        reply, trace = self.handle(conv, message)
        candidates = self.pending(conv)
        conversation = repo.get_conversation(conv)

        return (
            conv,
            reply,
            bool(conversation["closed"]),
            candidates,
            trace,
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

    def change_candidate_kind(self, candidate_id: int, kind: str):
        """Change a pending candidate's kind and re-derive judge state.

        Returns the conversation id. Raises ValueError for a bad kind,
        KeyError if missing, ValueError if not pending.
        """
        if kind not in KINDS:
            raise ValueError("Kind must be note, idea, or task")

        candidate = repo.get_candidate(candidate_id)
        if candidate is None:
            raise KeyError("candidate not found")
        if candidate["status"] != "pending":
            raise ValueError("candidate not pending")

        # Reset judge state and set the new kind in one update.
        repo.update_candidate(
            candidate_id,
            kind=kind,
            op="new",
            item_id=None,
            proposed_text=None,
            question=None,
        )

        # Re-judge tasks so dedup/update detection stays consistent.
        if kind == "task":
            self.judge(candidate_id)

        return candidate["conv_id"]

    def change_candidate_topic(self, candidate_id: int, topic: str):
        """Change a pending candidate's topic.

        A new topic is normalized (trimmed, title-cased) and registered.
        A malformed one — empty or over TOPIC_MAX_WORDS words — falls back to
        INBOX rather than blocking the turn. Returns (conv_id, applied_topic),
        where applied_topic is INBOX on fallback. Raises KeyError if missing,
        ValueError if not pending. Topic doesn't affect judge state.
        """
        candidate = repo.get_candidate(candidate_id)
        if candidate is None:
            raise KeyError("candidate not found")
        if candidate["status"] != "pending":
            raise ValueError("candidate not pending")

        cleaned = (topic or "").strip()
        if not cleaned or len(cleaned.split()) > TOPIC_MAX_WORDS:
            applied = "INBOX"
        else:
            applied = cleaned.title()
            # Register a new topic (case-insensitive) for the UI filter and extraction.
            existing = self.get_topics()
            if applied != "INBOX" and applied.casefold() not in {
                t.casefold() for t in existing
            }:
                self.set_topics(existing + [applied])

        repo.update_candidate(candidate_id, topic=applied)
        return candidate["conv_id"], applied

    def force_new_candidate(self, candidate_id: int):
        """Override the judge's match and save the candidate as a new item.

        The user's "no, this is a separate task" escape hatch for a candidate
        the judge tied to an existing one. Resets judge state to `new` and
        confirms, creating a fresh item. Returns (conv_id, closed) like
        ui_resolve; raises KeyError if not pending.
        """
        candidate = repo.get_candidate(candidate_id)
        if not candidate or candidate["status"] != "pending":
            raise KeyError("candidate not pending")

        # Reset the judged match, then confirm as a brand-new item.
        repo.update_candidate(
            candidate_id,
            op="new",
            item_id=None,
            proposed_text=None,
            question=None,
        )
        fresh = repo.get_candidate(candidate_id)
        self.confirm(fresh)

        remaining = self.pending(candidate["conv_id"])
        closed = not remaining
        if closed:
            self.close(candidate["conv_id"])
        return candidate["conv_id"], closed

    def get_pending(self, conv_id):
        return self.pending(conv_id)

    def get_messages(self, conv_id, limit=100):
        """Chat history, oldest first — for rehydrating the UI after a refresh."""
        return repo.get_history(conv_id, limit=limit)

    def get_conversation(self, conv_id):
        """The conversation row (or None), for rehydration/closed checks."""
        return repo.get_conversation(conv_id)

    def get_latest_open_conversation(self):
        """Most recent open conversation with pending items, or None.

        Lets the UI resume a review after losing session state.
        """
        return repo.get_latest_open_conversation()

    def get_items(self, kind=None, status=None):
        return repo.get_items(kind=kind, status=status)

    def get_tasks(self):
        return repo.get_open_tasks()

    def get_topics(self):
        return repo.get_topics()

    def set_topics(self, topics):
        """Save the user's topic list.

        Entries over TOPIC_MAX_WORDS words are rejected so the editor can
        surface it. (The agentic change_topic path instead falls back to INBOX.)
        """
        too_long = [
            t for t in topics
            if t.strip() and len(t.strip().split()) > TOPIC_MAX_WORDS
        ]
        if too_long:
            raise ValueError(
                f"Topics must be at most {TOPIC_MAX_WORDS} words. "
                f"Too long: {', '.join(too_long)}"
            )
        repo.set_topics(topics)

    def find_similar_tasks(self, text, topic=None, n_results=None):
        """Retrieve candidate saved tasks for the judge.

        Returns at most `retrieval_top_k` matches, dropping any below
        `retrieval_min_similarity` so clearly-unrelated tasks never reach the
        judge. Both default from config (top-k 4, min-similarity 0.3) and match
        the judge-RAG eval.
        """
        if n_results is None:
            n_results = settings.retrieval_top_k

        matches = self.retrieval.search_tasks(
            text=text,
            topic=topic,
            n_results=n_results,
        )
        return [
            m for m in matches
            if m.get("similarity", 0) >= settings.retrieval_min_similarity
        ]

    def index_task(self, task_id, text, topic, status="open"):
        """Add or refresh a task in the vector index.

        Failures are logged, not raised, so indexing never blocks confirmation.
        """
        try:
            self.retrieval.index_task(task_id, text, topic, status)
        except Exception:
            log.warning(
                "Failed to index task %s in vector store", task_id,
                exc_info=True,
            )

    def reindex_status(self, task_id, status):
        """Update a task's status in the vector index (best effort)."""
        try:
            self.retrieval.update_metadata(
                str(task_id),
                {"status": status},
            )
        except Exception:
            log.warning(
                "Failed to update index status for task %s", task_id,
                exc_info=True,
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
        # Track every LLM call this turn makes. The finally drains them even if
        # the turn raises, so partial calls are persisted and never leak into
        # the next turn's log.
        llm.start_tracking()
        try:
            self.say(conv, "user", text)

            pend = self.pending(conv)
            log.info(
                "handle conv=%s msg_chars=%d pending=%d -> %s",
                conv,
                len(text or ""),
                len(pend),
                "interpret_flow" if pend else "process_dump (no pending)",
            )
            lines = []
            other = text
            new = None
            trace = None

            if pend:
                other, lines, trace = self.interpret_flow(conv, pend, text)
                log.info(
                    "handle after interpret: lines=%s other_chars=%d",
                    lines,
                    len(other or ""),
                )

            if other:
                new = self.process_dump(conv, other)

                if new is None:
                    log.info("process_dump: not a brain dump; no candidates created")
                    lines.append(TEMPLATE)
                elif not new:
                    log.info("process_dump: brain dump but no items extracted")
                    lines.append("I couldn't find any items in that.")
                else:
                    log.info("process_dump: created %d new candidate(s) %s", len(new), new)

            reply = self.compose(conv, lines, new)

            if pend and not self.pending(conv):
                self.close(conv)
                reply += (
                    "\n\n✅ All items resolved — this chat is finished. "
                    "Send a new message to start another."
                )

            self.say(conv, "assistant", reply)
            return reply, trace
        finally:
            self._persist_tracked_calls()

    def _persist_tracked_calls(self):
        """Drain tracked LLM calls and store them. Best-effort: never raises.

        A persistence failure must not break a pipeline turn, so it degrades to
        a log line while the turn's original result (or exception) propagates.
        """
        try:
            entries = llm.pop_calls()
            if not entries:
                return
            rows = [metrics.record_to_row(e) for e in entries]
            repo.insert_llm_calls(rows)
        except Exception:
            log.warning("Failed to persist LLM call metrics", exc_info=True)

    def list_metrics(self, step=None, model=None, status=None, limit=100):
        return repo.list_llm_calls(step=step, model=model, status=status, limit=limit)

    def metrics_summary(self, step=None, model=None):
        return repo.summarize_llm_calls(step=step, model=model)

    # ---------- Extract brain dumps ----------

    def process_dump(self, conv, text):
        """Return None for a non-dump, otherwise return candidate IDs."""
        topics = self.get_topics() + ["INBOX"]

        if not llm.is_brain_dump(text):
            return None

        # Extract items, then classify each into a kind — two separate calls.
        items = llm.extract(text, topics)
        classified = llm.classify(items)

        if len(classified) != len(items):
            raise ValueError(
                f"classify returned {len(classified)} items "
                f"for {len(items)} inputs"
            )

        for item, result in zip(items, classified):
            item["text"] = result["text"]
            item["kind"] = result["kind"]

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

            if operation == "needs_human" and not result.get("clarifying_question"):
                raise ValueError("Missing clarification question")

            if operation == "new" or task_id not in similar_ids:
                task_id = None

            repo.update_candidate(
                    candidate_id,
                    op=operation,
                    item_id=task_id,
                    proposed_text=result.get("proposed_text"),
                    question=(
                        result.get("clarifying_question")
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
                # Index so future dumps can detect this as a duplicate/update.
                self.index_task(
                    item_id,
                    candidate["text"],
                    candidate["topic"],
                )
                message = "task created"

            elif operation == "update":
                if item_id is None:
                    raise ValueError("Missing item ID for task update")

                new_text = candidate.get("proposed_text") or candidate["text"]
                repo.update_task_text(item_id, new_text)
                # Re-index with the revised text.
                self.index_task(item_id, new_text, candidate["topic"])
                message = "task updated"

            elif operation == "complete":
                if item_id is None:
                    raise ValueError("Missing item ID for task completion")

                repo.set_task_status(item_id, "done")
                # Drop from the open-task index so it stops matching.
                self.reindex_status(item_id, "done")
                message = "task completed"

            elif operation == "cancel":
                if item_id is None:
                    raise ValueError("Missing item ID for task cancellation")

                repo.set_task_status(item_id, "cancelled")
                self.reindex_status(item_id, "cancelled")
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
            # New text invalidates the prior judge decision: reset judge state
            # and clear the stale question, then re-judge. Otherwise a task
            # keeps a question that no longer matches and can never confirm.
            repo.update_candidate(
                candidate_id,
                text=text.strip(),
                op="new",
                item_id=None,
                proposed_text=None,
                question=None,
            )

            if candidate["kind"] == "task":
                self.judge(candidate_id)

        else:
            raise ValueError("bad action")

        remaining = self.pending(candidate["conv_id"])
        closed = not remaining

        if closed:
            self.close(candidate["conv_id"])

        return candidate["conv_id"], closed

    # ---------- Interpret replies to pending candidates ----------

    def apply_plan(self, plan, candidates) -> list[str]:
        """Apply a validated SubmitPlan to pending candidates.

        A failing action is logged and skipped (candidate stays pending) so it
        never crashes the turn. Each candidate is acted on at most once.
        """
        by_no = {c["display_no"]: c for c in candidates}
        done = set()
        output = []

        for action in plan.actions:
            candidate = by_no.get(action.display_no)
            if not candidate or candidate["id"] in done:
                continue

            op = action.operation
            text = (action.text or "").strip()

            try:
                if op == "confirm":
                    output.append(self.confirm(candidate))

                elif op == "reject":
                    # Agreeing a candidate is a duplicate == reject.
                    output.append(self.reject(candidate))

                elif op == "change_text" and text:
                    # New text invalidates the prior judge decision (see
                    # ui_resolve): reset judge state, clear the stale question,
                    # re-judge. Else it confirms against the wrong task.
                    repo.update_candidate(
                        candidate["id"],
                        text=text,
                        op="new",
                        item_id=None,
                        proposed_text=None,
                        question=None,
                    )
                    if candidate["kind"] == "task":
                        self.judge(candidate["id"])
                    output.append(f"#{candidate['display_no']}: text updated")

                elif op == "change_kind" and action.kind:
                    self.change_candidate_kind(candidate["id"], action.kind)
                    output.append(
                        f"#{candidate['display_no']}: kind -> {action.kind}"
                    )

                elif op == "change_topic" and action.topic:
                    _, applied = self.change_candidate_topic(
                        candidate["id"], action.topic
                    )
                    if applied == "INBOX" and action.topic.strip().title() != "INBOX":
                        output.append(
                            f"#{candidate['display_no']}: couldn't use "
                            f"'{action.topic}' as a topic (too long) — "
                            f"filed under INBOX"
                        )
                    else:
                        output.append(
                            f"#{candidate['display_no']}: topic -> {applied}"
                        )

                elif op == "needs_human" and action.question:
                    repo.update_candidate(
                        candidate["id"],
                        op="needs_human",
                        question=action.question,
                    )
                    output.append(
                        f"#{candidate['display_no']}: needs your input"
                    )

                else:
                    # Invalid actions leave the candidate pending.
                    continue

            except Exception:
                log.warning(
                    "apply_plan action failed; leaving candidate pending",
                    exc_info=True,
                )
                continue

            done.add(candidate["id"])

        return output

    def _apply_fast_path(self, candidates, plan):
        """Apply a deterministic fast-path plan; return summary lines."""
        by_no = {c["display_no"]: c for c in candidates}
        method = self.confirm if plan.action == "confirm" else self.reject

        lines = [
            method(by_no[display_no])
            for display_no in plan.targets
            if display_no in by_no
        ]

        # Tell the user which items were left because they need an answer.
        for display_no in plan.skipped_questioned:
            lines.append(
                f"#{display_no}: still needs an answer before it can be "
                f"{plan.action}ed"
            )

        return lines

    def interpret_flow(self, conv, candidates, text):
        """Return (other content, summary lines, trace).

        `trace` (see `_build_trace`) is always returned so the UI can show the
        per-candidate decisions, independent of SHOW_INTERPRET_TRACE.
        """
        # Deterministic fast path: resolve simple, unambiguous replies
        # ("confirm all", "reject #2", "yes to 1 and 3") without the LLM.
        plan = fast_path.parse(text, candidates)
        if plan is not None:
            log.info(
                "interpret_flow: fast_path %s targets=%s",
                plan.action,
                plan.targets,
            )
            lines = self._apply_fast_path(candidates, plan)
            trace = {
                "source": "fast_path",
                "decisions": [
                    {
                        "display_no": display_no,
                        "operation": plan.action,
                        "before_text": None,
                        "after_text": None,
                        "kind": None,
                        "topic": None,
                        "question": None,
                    }
                    for display_no in plan.targets
                ],
                "new_dump": "",
                "tools": [],
                "steps": [],
            }
            return "", lines, trace

        try:
            # Re-read pending state for this turn. The >20 cap is checked
            # only here (agentic path), after the fast-path short-circuit.
            pending = repo.get_pending_candidates(conv)
            if len(pending) > 20:
                return (
                    "",
                    [
                        "You have more than 20 pending items — please resolve "
                        "some before adding more."
                    ],
                    _build_trace(None, pending, "none"),
                )

            log.info(
                "interpret_flow: agent path prompt=%s model=%s pending=%d",
                llm.get_prompt_for_operation("interpret"),
                self.agent.model,
                len(pending),
            )
            result = self.agent.run(
                conv, pending, text, topics=self.get_topics()
            )

            if result.plan is not None:
                log.info(
                    "interpret_flow: plan actions=%s new_dump_chars=%d tools=%s",
                    [(a.display_no, a.operation) for a in result.plan.actions],
                    len((result.plan.new_dump or "")),
                    result.tools_used,
                )

            if result.plan is None:
                # max_steps exhausted with no submit_plan: safe fallback.
                return (
                    "",
                    _with_trace(
                        [
                            "Sorry, I couldn't interpret that. Try "
                            "`confirm all`, `reject #2`, or use the buttons."
                        ],
                        result,
                    ),
                    _build_trace(result, pending, "agent"),
                )

            # Snapshot pending for the trace BEFORE apply_plan mutates in
            # place, so `before_text` reflects the pre-plan state.
            pending = repo.get_pending_candidates(conv)
            trace = _build_trace(result, pending, "agent")
            lines = self.apply_plan(result.plan, pending)

            # new_dump flows back out so handle() feeds it to process_dump as
            # new candidates. `unrelated` is dropped: filler, not capturable.
            other = (result.plan.new_dump or "").strip()

            return other, _with_trace(lines, result), trace

        except Exception:
            log.warning("Reply interpretation failed", exc_info=True)
            return (
                "",
                [
                    "Sorry, I couldn't interpret that. Try `confirm all`, "
                    "`reject #2`, or use the buttons."
                ],
                {
                    "source": "none",
                    "decisions": [],
                    "new_dump": "",
                    "tools": [],
                    "steps": [],
                },
            )

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

        # Keep the index in sync so only open tasks are searchable. Reopening
        # re-adds the text in case the index entry was dropped.
        if status == "open":
            self.index_task(task_id, task["text"], task["topic"], "open")
        else:
            self.reindex_status(task_id, status)

        return {"id": task_id, "status": status}
