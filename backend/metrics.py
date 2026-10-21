"""Pure mapping from an in-memory LLM-call record to a persistable row.

Kept DB- and framework-free so it is a clean unit/property-test target.
"""


def record_to_row(entry: dict) -> dict:
    """Map one `_record_call()` entry (backend/llm.py) to an `llm_calls` row.

    Token and cost fields pass through under their storage names; `status` and
    the integer `error` flag are both derived from the boolean `error` so the
    two stay consistent. `created_at` is omitted — the DB default supplies it.
    """
    is_error = bool(entry["error"])
    return {
        "step": entry["step"],
        "model": entry["model"],
        "input_tokens": entry["prompt_tokens"],
        "output_tokens": entry["completion_tokens"],
        "cached_tokens": entry["cached_tokens"],
        # latency is tracked in seconds but stored/served in milliseconds
        "latency_ms": round(entry["latency_s"] * 1000),
        # None means the model had no price entry; keep it null at storage time
        # (aggregation collapses null to zero, not the mapping)
        "input_cost_usd": entry["input_cost_usd"],
        "output_cost_usd": entry["output_cost_usd"],
        "cost_usd": entry["cost_usd"],
        "status": "error" if is_error else "ok",
        "error": 1 if is_error else 0,
    }
