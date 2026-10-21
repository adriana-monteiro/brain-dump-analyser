"""Agentic interpret loop — turns a user reply into a validated plan.

A bounded tool-calling loop over brain-dump candidates. It finishes by calling
the terminal `submit_plan` tool, whose strict arguments are the structured
output. Each step appends a `trace` entry and routes every model call through
`llm._record_call`, so agentic turns share the one-shot steps' latency/token/
cost log (active only when `llm.start_tracking()` is on).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from backend import llm
from backend import repositories as repo
from backend.schemas.llm import GetHistoryArgs, SubmitPlan


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_history",
        "description": (
            "Fetch the conversation history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": (
                        "How many recent turns to fetch (1-10)."
                    ),
                }
            },
            "required": ["n"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "submit_plan",
        "description": (
            "Submit the final resolution plan for the "
            "pending candidates. Calling this ends the turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "display_no": {"type": "integer"},
                            "operation": {
                                "type": "string",
                                "enum": [
                                    "confirm",
                                    "reject",
                                    "change_text",
                                    "change_kind",
                                    "change_topic",
                                    "needs_human",
                                ],
                            },
                            "text": {"type": ["string", "null"]},
                            "kind": {
                                "type": ["string", "null"],
                                "enum": ["note", "idea", "task", None],
                            },
                            "topic": {"type": ["string", "null"]},
                            "question": {"type": ["string", "null"]},
                        },
                        "required": [
                            "display_no",
                            "operation",
                            "text",
                            "kind",
                            "topic",
                            "question",
                        ],
                        "additionalProperties": False,
                    },
                },
                "new_dump": {"type": ["string", "null"]},
                "unrelated": {"type": ["string", "null"]},
            },
            "required": ["actions", "new_dump", "unrelated"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def _load_interpret_instructions() -> str:
    # Resolve via STEP_PROMPTS so the prompt version lives in one table.
    return llm.build_step_instructions("interpret").strip()


INTERPRET_INSTRUCTIONS = _load_interpret_instructions()


def _sampling_kwargs(model: str) -> dict[str, Any]:
    """Temperature only for non-reasoning models (o1/o3/o4/gpt-5 reject it)."""
    if model.startswith(("o1", "o3", "o4", "gpt-5")):
        return {}
    return {"temperature": 0}


def build_turn_context(
    message: str,
    candidates: list[dict[str, Any]],
    topics: list[str] | None = None,
) -> str:
    """Serialize the turn context for the model.

    Trims each candidate to the keys the model reasons over. `available_topics`
    is passed because topics are user-configurable, not a fixed enum.
    """
    trimmed = [
        {
            k: c.get(k)
            for k in (
                "display_no",
                "kind",
                "text",
                "topic",
                "op",
                "proposed_text",
                "question",
                "existing_item_text",
                "existing_item_status",
            )
        }
        for c in candidates
    ]
    return json.dumps(
        {
            "message": message,
            "pending_candidates": trimmed,
            "available_topics": topics or [],
        },
        ensure_ascii=False,
    )


@dataclass(frozen=True)
class InterpretResult:
    """Outcome of one interpret turn.

    `plan` is None when the loop ended without a valid terminal call (stall or
    max_steps exhaustion). `tools_used` lists tool names in call order (with
    `"none"` for no-call/exhausted steps). `trace` is the per-step log.
    """

    plan: SubmitPlan | None
    tools_used: list[str]
    trace: list[dict[str, Any]]


class InterpretAgent:
    """Own the bounded tool-calling loop that turns a reply into a plan."""

    def __init__(self, client: Any = None, model: str | None = None) -> None:
        self.client = client or OpenAI()
        # Interpret's model lives in the central STEP_MODELS map (default
        # gpt-4.1), honoring MODEL_INTERPRET / INTERPRET_MODEL overrides.
        self.model = model or llm.model_for_step("interpret")

    def run(
        self,
        conv_id: Any,
        candidates: list[dict[str, Any]],
        message: str,
        max_steps: int = 4,
        topics: list[str] | None = None,
    ) -> InterpretResult:
        if not message.strip():
            raise ValueError("message cannot be empty.")
        if max_steps < 1 or max_steps > 8:
            raise ValueError("max_steps must be between 1 and 8.")

        model_input: list[Any] = [
            {
                "role": "user",
                "content": build_turn_context(message, candidates, topics),
            }
        ]
        trace: list[dict[str, Any]] = []

        for step in range(1, max_steps + 1):
            t0 = time.perf_counter()
            response = None
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=INTERPRET_INSTRUCTIONS,
                    input=model_input,
                    tools=TOOLS,
                    parallel_tool_calls=True,
                    store=False,
                    **_sampling_kwargs(self.model),
                )
                llm._record_call("interpret", self.model, t0, response)
            except Exception:
                llm._record_call("interpret", self.model, t0, None)
                raise

            model_input.extend(response.output)
            calls = [item for item in response.output if item.type == "function_call"]
            
            if not calls:
                trace.append(self._trace(step, "none", "no_tool_calls"))
                continue

            for call in calls:
                if call.name == "submit_plan":
                    plan = self._validate_submit_plan(call)
                    if isinstance(plan, SubmitPlan):
                        trace.append(self._trace(step, call.name, "ok"))
                        return InterpretResult(
                            plan=plan,
                            tools_used=[t["tool"] for t in trace],
                            trace=trace,
                        )
                    # plan is the structured _tool_error dict on failure.
                    trace.append(
                        self._trace(
                            step,
                            call.name,
                            "error",
                            detail=plan.get("error", {}).get("message"),
                        )
                    )
                    model_input.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(plan),
                        }
                    )
                else:
                    result = self.execute_tool_call(conv_id, call)
                    ok = bool(result.get("ok"))
                    trace.append(
                        self._trace(
                            step,
                            call.name,
                            "ok" if ok else "error",
                            args=self._safe_args(call),
                            detail=(
                                None
                                if ok
                                else result.get("error", {}).get("message")
                            ),
                        )
                    )
                    model_input.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(result),
                        }
                    )

        # max_steps hit with no submit_plan
        trace.append(self._trace(max_steps, "none", "exhausted"))
        return InterpretResult(
            plan=None,
            tools_used=[t["tool"] for t in trace],
            trace=trace,
        )

    def execute_tool_call(self, conv_id: Any, call: Any) -> dict[str, Any]:
        try:
            arguments = json.loads(call.arguments)
        except (TypeError, json.JSONDecodeError):
            return self._tool_error("invalid_arguments", "Arguments are not valid JSON.")
        if not isinstance(arguments, dict):
            return self._tool_error("invalid_arguments", "Arguments must be an object.")

        handlers = {"get_history": self._get_history}
        handler = handlers.get(call.name)
        if handler is None:
            return self._tool_error("unknown_tool", f"Unknown tool: {call.name}")
        try:
            return handler(conv_id, arguments)
        except ValueError as exc:
            return self._tool_error("invalid_arguments", str(exc))
        except Exception:
            return self._tool_error(
                "execution_failed", "The tool failed. Proceed or explain the failure."
            )

    def _get_history(self, conv_id: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        args = GetHistoryArgs(**arguments)
        n = max(1, min(args.n, 10))
        return {"ok": True, "turns": repo.get_history(conv_id, limit=n)}

    def _validate_submit_plan(self, call: Any) -> Any:
        try:
            return SubmitPlan(**json.loads(call.arguments))
        except (TypeError, json.JSONDecodeError, ValidationError) as exc:
            return self._tool_error("invalid_arguments", str(exc))

    @staticmethod
    def _tool_error(error_type: str, message: str) -> dict[str, Any]:
        return {"ok": False, "error": {"type": error_type, "message": message}}

    @staticmethod
    def _trace(
        step: int,
        tool: str,
        status: str,
        args: dict[str, Any] | None = None,
        detail: str | None = None,
    ) -> dict[str, Any]:
        """One structured trace entry.

        `status` is ok / error / no_tool_calls / exhausted. `args` and `detail`
        are omitted when empty so the common ok entry stays compact.
        """
        entry: dict[str, Any] = {"step": step, "tool": tool, "status": status}
        if args:
            entry["args"] = args
        if detail:
            entry["detail"] = detail
        return entry

    @staticmethod
    def _safe_args(call: Any) -> dict[str, Any] | None:
        """Parsed tool-call arguments for the trace, or None if unparseable."""
        try:
            parsed = json.loads(call.arguments)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None
