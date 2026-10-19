"""LLM layer: prompt loading (prompts/*.txt) + provider client (OpenAI Responses API)."""
import json
import logging
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Literal, Optional, get_args

from pydantic import create_model

from backend.schemas.llm import SCHEMAS
from backend.llm_pricing import cost_parts

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(
    os.getenv("PROMPTS_DIR", Path(__file__).resolve().parent / "prompts")
)

# Pipeline step -> prompt file in prompts/ (without .txt).
STEP_PROMPTS = {
    "dump": "dump_v1",
    "extract": "extract_v1",
    "extract_classify": "extract_classify_v1",
    "classify": "classify_v1",
    "judge": "task_judge_v1",
    "interpret": "interpret_v1",
}

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1500"))
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "2"))


# ---------------- prompts ----------------

def get_prompt_for_operation(step: str) -> str:
    """PROMPT_<STEP>=<file> overrides the default prompt."""
    if step not in STEP_PROMPTS:
        raise ValueError(f"Unknown pipeline step: {step}")
    return os.getenv(f"PROMPT_{step.upper()}") or STEP_PROMPTS[step]


def build_instructions(prompt_file: str, **context) -> str:
    """Load prompts/<prompt_file>.txt and substitute $placeholders."""
    root = PROMPTS_DIR.resolve()
    path = (root / f"{prompt_file}.txt").resolve()

    if not path.is_relative_to(root):
        raise ValueError(f"Invalid prompt path: {prompt_file}")
    if not path.is_file():
        raise ValueError(f"No prompt file {path} (prompt: {prompt_file})")

    return Template(path.read_text(encoding="utf-8")).substitute(**context)


def build_step_instructions(step: str, **context) -> str:
    """Load the prompt configured for a pipeline step."""
    return build_instructions(get_prompt_for_operation(step), **context)


# ---------------- schemas ----------------

def _base_item_model(step: str):
    """Item model from the static schema (SCHEMAS[step].items: list[Item])."""
    annotation = SCHEMAS[step].model_fields["items"].annotation
    return get_args(annotation)[0]


@lru_cache(maxsize=256)
def _extract_schema(step: str, topics: tuple):
    """Copy of SCHEMAS[step] whose topic fields only accept the given topics.

    Subclassing the existing Item keeps every other field (text, kind, ...)
    exactly as defined in backend/schemas/llm.py.
    """
    Topic = Literal[topics]  # equivalent to Literal["Work", "INBOX", ...]
    Item = create_model(
        "Item",
        __base__=_base_item_model(step),
        topic=(Topic, ...),
        alternative_topic=(Optional[Topic], None),
    )
    return create_model("ExtractOut", items=(list[Item], ...))


def extract_schema(step: str, topics) -> type:
    unique = tuple(dict.fromkeys(topics))  # dedupe, keep order
    if not unique:
        raise ValueError("topics must not be empty")
    return _extract_schema(step, unique)


# ---------------- call tracking ----------------

# Per-thread call log, enabled only when start_tracking() is called.
_tls = threading.local()


def start_tracking():
    _tls.calls = []


def pop_calls() -> list[dict]:
    calls = getattr(_tls, "calls", None) or []
    _tls.calls = []
    return calls


def _record_call(step, model, t0, response=None):
    """Append one entry to the per-thread log (no-op unless tracking)."""
    calls = getattr(_tls, "calls", None)
    if calls is None:
        return

    usage = getattr(response, "usage", None) if response is not None else None
    entry = {
        "step": step,
        "model": (getattr(response, "model", None) or model),
        "latency_s": time.perf_counter() - t0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "input_cost_usd": None,
        "cached_input_cost_usd": None,
        "output_cost_usd": None,
        "cost_usd": None,
        "error": response is None,
    }

    if usage is not None:
        prompt_tokens = usage.input_tokens or 0
        completion_tokens = usage.output_tokens or 0
        details = getattr(usage, "input_tokens_details", None)
        cached_tokens = (getattr(details, "cached_tokens", 0) or 0) if details else 0

        entry.update(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )

        parts = cost_parts(entry["model"], prompt_tokens, completion_tokens, cached_tokens)
        if parts is not None:
            input_cost, cached_cost, output_cost = parts
            entry.update(
                input_cost_usd=input_cost,
                cached_input_cost_usd=cached_cost,
                output_cost_usd=output_cost,
                cost_usd=input_cost + cached_cost + output_cost,
            )

    calls.append(entry)


# ---------------- client ----------------

def _refusal_text(response):
    for output_item in response.output:
        if output_item.type == "message":
            for item in output_item.content:
                if item.type == "refusal":
                    return item.refusal
    return None


class LLMClient:
    """OpenAI Responses API wrapper: step name in, validated dict out."""

    def __init__(self, model: str | None = None):
        import openai

        self.model = model or DEFAULT_MODEL
        self._client = openai.OpenAI()

    def _sampling_kwargs(self) -> dict:
        # Reasoning models reject `temperature`.
        if self.model.startswith(("o1", "o3", "o4", "gpt-5")):
            return {}
        return {"temperature": 0}

    def _call_once(self, step, instructions, payload, schema) -> dict:
        t0 = time.perf_counter()
        response = None
        try:
            log.debug("step=%s instructions=%s", step, instructions)
            log.debug("step=%s payload=%s", step, payload)
            response = self._client.responses.parse(
                model=self.model,
                instructions=instructions,
                input=payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False),
                text_format=schema,
                max_output_tokens=MAX_OUTPUT_TOKENS,  # stops runaway generations
                store=False,
                **self._sampling_kwargs(),
            )

            if response.status == "incomplete":
                raise RuntimeError(f"Incomplete response: {response.incomplete_details}")

            parsed = response.output_parsed
            if parsed is None:
                refusal = _refusal_text(response)
                raise RuntimeError(
                    f"Model refused or returned no parsed output: {refusal or 'no parsed output'}"
                )

            _record_call(step, self.model, t0, response)
            return parsed.model_dump()
        except Exception:
            # Log failed attempts too, with usage if we got a response back.
            _record_call(step, self.model, t0, None)
            raise

    def run_step(self, step: str, payload, *, schema=None, **context) -> dict:
        """Load the step prompt, call the Responses API (with retry), return a dict."""
        instructions = build_step_instructions(step, **context)
        schema = schema or SCHEMAS[step]

        last_exc = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self._call_once(step, instructions, payload, schema)
            except Exception as exc:
                last_exc = exc
                log.warning("LLM step %s failed (attempt %d/%d): %s", step, attempt, MAX_ATTEMPTS, exc)
        raise last_exc


_llm: LLMClient | None = None


def get_client() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def run_step(step, payload, **context):
    return get_client().run_step(step, payload, **context)


# ---------------- public API (one function per LLM box in the diagram) ----------------

def is_brain_dump(text):
    return bool(run_step("dump", {"message": text}).get("is_brain_dump"))


def _clean_alternatives(items):
    """Drop alternative_topic when it is INBOX or the same as topic."""
    for it in items:
        if it.get("alternative_topic") in ("INBOX", it["topic"]):
            it["alternative_topic"] = None
    return items


def extract(text, topics):
    items = run_step(
        "extract",
        text,
        schema=extract_schema("extract", topics),
        topics=", ".join(topics)
    )["items"]
    return _clean_alternatives(items)


def extract_classify(text, topics):
    items = run_step(
        "extract_classify",
        {"message": text, "topics": list(topics)},
        schema=extract_schema("extract_classify", topics),
        topics=", ".join(topics),
    )["items"]
    return _clean_alternatives(items)


def classify(items):
    result = run_step(
        "classify",
        payload=json.dumps(items, ensure_ascii=False),
    )
    log.debug("classify input=%r result=%r", items, result)
    return result["items"]


def judge(task, similar):
    return run_step("judge", {"task": task, "similar": similar})


def interpret(candidates, history, message):
    return run_step(
        "interpret",
        {
            "candidates": candidates,
            "history": history,
            "message": message,
        },
    )