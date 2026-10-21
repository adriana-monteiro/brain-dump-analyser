
"""Eval runner.

Examples:
    uv run --env-file .env python -m evals.run
    uv run --env-file .env python -m evals.run --step judge -v
    uv run --env-file .env python -m evals.run --model gpt-4o --json out/gpt4o.json --fail-under 0.8

Run the two-call extract+classify split:
    uv run --env-file .env python -m evals.run --step extract_split

Cases live in evals/cases/<step>.json.
"""
import argparse
import difflib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend import llm, llm_pricing, retrieval
from backend.config import settings
from backend.interpret_agent import InterpretAgent

judge_rag_retrieval = retrieval.RetrievalService(
    persist_directory="evals/data/chroma_judge_rag",
    collection_name="judge_rag_tasks",
)

# Retrieval-only step reads from the corpus built by evals/run_retrieval.py.
retrieval_eval = retrieval.RetrievalService(
    persist_directory="evals/data/chroma_retrieval",
    collection_name="retrieval_benchmark_tasks",
)

CASES = Path(__file__).parent / "cases"
# Directories searched for case files when a name is given to --cases (or a
# default step name is used). cases/ is searched first, then the per-step
# corpus folders kept alongside it (e.g. evals/01_brain_dump/,
# evals/02_extract/), discovered by the NN_name numbering convention so new
# step folders are picked up without editing this list.
EVALS_ROOT = Path(__file__).parent


def case_dirs():
    """cases/ first, then numbered per-step folders (01_*, 02_*, ...)."""
    numbered = sorted(
        d for d in EVALS_ROOT.iterdir()
        if d.is_dir() and len(d.name) > 3 and d.name[:2].isdigit()
        and d.name[2] == "_"
    )
    return [CASES, *numbered]


def find_case_file(name):
    """Return the first <dir>/<name>.json across case_dirs(), else raise."""
    dirs = case_dirs()
    for directory in dirs:
        path = directory / f"{name}.json"
        if path.is_file():
            return path
    searched = ", ".join(str(d) for d in dirs)
    raise FileNotFoundError(f"No case file {name}.json in: {searched}")

LLM_STEPS = {
    "dump": ["dump"],
    "extract": ["extract"],
    "classify": ["classify"],
    "extract_split": ["extract", "classify"],
    "judge": ["judge"],
    # The interpret step drives the agentic InterpretAgent loop, which carries
    # its instructions in code (INTERPRET_INSTRUCTIONS). There is no prompt
    # file or STEP_PROMPTS entry for it, so report no prompt (like retrieval).
    "interpret": [],
    "retrieval": [],  # No LLM prompt
    "judge_rag": ["judge"],
}

CASE_FILES = {
    "extract": "extract",
    "extract_split": "extract",
    "judge_rag": "task_judge_rag",
}

DEFAULT_TOPICS = ["Work", "Personal", "Health", "Finance"]
ID_OPS = {"update", "duplicate", "complete", "cancel"}


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else f"{a}/{b}"


def percentile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def money(x):
    return "n/a" if x is None else (f"${x:.4f}" if x < 0.1 else f"${x:.2f}")


def call_cost_parts(call):
    """Return input, cached-input, and output costs for one call."""
    if call.get("error"):
        return None
    return llm_pricing.cost_parts(
        call["model"],
        call["prompt_tokens"],
        call["completion_tokens"],
        call.get("cached_tokens", 0),
    )


def cost_split(us):
    """Sum costs for usage records with known prices."""
    ok = [u for u in us if u["cost_usd"] is not None]
    if not ok:
        return None

    return (
        sum(u["cost_usd"] for u in ok),
        sum(u["cost_input_usd"] + u["cost_cached_input_usd"] for u in ok),
        sum(u["cost_output_usd"] for u in ok),
    )


def usage_line(results):
    """Latency, token, and cost summary for case results."""
    usages = [r["usage"] for r in results]
    if not any(u["calls"] for u in usages):
        return "no LLM calls recorded"

    n = len(usages)
    costs = cost_split(usages)

    if costs:
        cost = (
            f"cost {money(costs[0])} total = "
            f"{money(costs[1])} input + {money(costs[2])} output, "
            f"{money(costs[0] / n)}/case"
        )
    else:
        cost = "cost n/a"

    return (
        f"latency p50 {percentile([u['latency_s'] for u in usages], 50):.1f}s "
        f"p95 {percentile([u['latency_s'] for u in usages], 95):.1f}s · "
        f"{cost} · "
        f"{sum(u['prompt_tokens'] for u in usages) / n:,.0f} in / "
        f"{sum(u['completion_tokens'] for u in usages) / n:,.0f} "
        f"out tokens/case · "
        f"{sum(u['calls'] for u in usages) / n:.1f} calls/case"
    )


# ---------------------------------------------------------------------------
# Dump detection
# ---------------------------------------------------------------------------

def run_dump(c):
    return llm.is_brain_dump(c["input"])


def score_dump(c, out):
    want = c["expected"]
    return {
        "ok": out == want,
        "detail": f"expected {want}, got {out}",
        "counts": {
            "fp": int(bool(out) and not want),
            "fn": int(bool(want) and not out),
        },
    }


def sum_dump(t, n):
    return f"false positives {t['fp']}, false negatives {t['fn']}"


# ---------------------------------------------------------------------------
# Extraction
#
# extract:        text + topic
# extract_split:  extract followed by classify, two calls (text + topic + kind)
# ---------------------------------------------------------------------------

def _topics(c):
    return c.get("topics", DEFAULT_TOPICS) + ["INBOX"]


def run_extract(c):
    return llm.extract(c["input"], _topics(c))


def run_classify(c):
    # Input is already-extracted items; output is a list of kind labels.
    return llm.classify(c["input"])


def run_extract_split(c):
    items = llm.extract(c["input"], _topics(c))
    classified_items = llm.classify(items)

    if len(classified_items) != len(items):
        raise ValueError(
            f"classify returned {len(classified_items)} items for {len(items)} inputs"
        )

    # Attach the rewritten text and predicted kind back to each item
    for item, classified in zip(items, classified_items):
        item["text"] = classified["text"]  # Rewritten text following rules
        item["kind"] = classified["kind"]  # Classified label

    return items


def _match(expected, got, threshold=0.6):
    """Greedily match expected items to output items by text."""
    pairs = []
    used = set()

    # Match required items before optional items.
    order = sorted(
        range(len(expected)),
        key=lambda i: bool(expected[i].get("optional")),
    )

    for expected_idx in order:
        exp = expected[expected_idx]
        best_score, best_idx = 0, None

        for got_idx, item in enumerate(got):
            if got_idx in used:
                continue

            text = item["text"].lower()
            ratio = difflib.SequenceMatcher(
                None,
                exp["text"].lower(),
                text,
            ).ratio()

            if exp.get("keywords"):
                score = (
                    1 + ratio
                    if all(k.lower() in text for k in exp["keywords"])
                    else 0
                )
            else:
                score = ratio if ratio >= threshold else 0

            if score > best_score:
                best_score, best_idx = score, got_idx

        if best_idx is not None:
            used.add(best_idx)
            pairs.append((expected_idx, best_idx))

    return sorted(pairs)


def _val_ok(want, got):
    return want == "*" or (got in want if isinstance(want, list) else got == want)


def _check_links(c, out, pairs):
    """Check whether contextual facts attach to the intended task."""
    expected_to_output = dict(pairs)
    counts = Counter()
    problems = []

    for link in c.get("links", []):
        keywords = [k.lower() for k in link["keywords"]]
        targets = [
            expected_to_output[i]
            for i in link["to"]
            if i in expected_to_output
        ]

        def contains_fact(item):
            text = f'{item["text"]} {item.get("context") or ""}'.lower()
            return all(k in text for k in keywords)

        counts["link_n"] += 1

        if any(contains_fact(out[i]) for i in targets):
            counts["link_ok"] += 1
        elif any(
            all(k in (item.get("context") or "").lower() for k in keywords)
            for i, item in enumerate(out)
            if i not in targets
        ):
            counts["mislink"] += 1
            problems.append(f"{keywords} attached to the WRONG task")
        elif any(
            all(k in item["text"].lower() for k in keywords)
            for item in out
        ):
            counts["orphan"] += 1
            problems.append(
                f"{keywords} left as a standalone item, not in task context"
            )
        else:
            counts["lost"] += 1
            problems.append(f"{keywords} lost (in no item or context)")

    return counts, problems


def score_extraction(c, out, check_kind):
    """Shared extraction scorer; kind is checked only for combined/split runs."""
    expected = c["expected"]
    pairs = _match(expected, out)

    matched_expected = {e for e, _ in pairs}
    matched_output = {g for _, g in pairs}

    required = [
        i for i, item in enumerate(expected)
        if not item.get("optional")
    ]
    missing = [
        expected[i]["text"]
        for i in required
        if i not in matched_expected
    ]

    unmatched = [
        i for i in range(len(out))
        if i not in matched_output
    ]
    extra = [out[i]["text"] for i in unmatched]

    wrong = []
    kind_ok = kind_total = 0
    topic_ok = topic_total = 0

    for expected_idx, output_idx in pairs:
        exp = expected[expected_idx]
        item = out[output_idx]

        if check_kind and "kind" in exp:
            kind_total += 1
            if _val_ok(exp["kind"], item.get("kind")):
                kind_ok += 1
            else:
                wrong.append(
                    f'{item["text"]!r}: kind wanted {exp["kind"]}, '
                    f'got {item.get("kind")}'
                )

        if "topic" in exp:
            topic_total += 1
            if _val_ok(exp["topic"], item.get("topic")):
                topic_ok += 1
            else:
                wrong.append(
                    f'{item["text"]!r}: topic wanted {exp["topic"]}, '
                    f'got {item.get("topic")}'
                )

    link_counts, link_problems = _check_links(c, out, pairs)

    problems = []
    if missing:
        problems.append(f"missing {missing}")
    if extra:
        problems.append(f"extra {extra}")
    if wrong:
        problems.append("wrong " + " | ".join(wrong))
    if link_problems:
        problems.append("link " + " | ".join(link_problems))

    return {
        "ok": not problems,
        "detail": "; ".join(problems),
        "counts": {
            "exp": len(required),
            "got": len(out),
            "matched": len(pairs),
            "matched_req": len(required) - len(missing),
            "kind_ok": kind_ok,
            "kind_total": kind_total,
            "topic_ok": topic_ok,
            "topic_total": topic_total,
            **link_counts,
        },
    }


def score_extract(c, out):
    # The extraction-only step does not predict kind.
    return score_extraction(c, out, check_kind=False)





def score_extract_split(c, out):
    return score_extraction(c, out, check_kind=True)


def sum_extract(t, n):
    parts = [
        f"recall {pct(t['matched_req'], t['exp'])}",
        f"precision {pct(t['matched'], t['got'])}",
    ]

    if t["kind_total"]:
        parts.append(f"kind {pct(t['kind_ok'], t['kind_total'])}")
    else:
        parts.append("kind n/a")

    if t["topic_total"]:
        parts.append(f"topic {pct(t['topic_ok'], t['topic_total'])}")
    else:
        parts.append("topic n/a")

    if t["link_n"]:
        parts.append(
            f"link {pct(t['link_ok'], t['link_n'])} "
            f"(mislink {t['mislink']}, orphan {t['orphan']}, lost {t['lost']})"
        )

    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Standalone kind classification
#
# Case input:    list of already-extracted items
# Case expected: list of kind strings, in the same order
# Example:       ["task", "note", "task"]
# ---------------------------------------------------------------------------


def score_classify(c, out):
    expected = c["expected"]
    predicted = out if isinstance(out, list) else []

    correct = 0
    missing = 0
    wrong = 0

    for i, exp in enumerate(expected):
        want = exp["kind"] if isinstance(exp, dict) else exp

        if i >= len(predicted):
            missing += 1
            continue

        item = predicted[i]
        got = item.get("kind") if isinstance(item, dict) else item

        if _val_ok(want, got):
            correct += 1
        else:
            wrong += 1

    extra = max(0, len(predicted) - len(expected))
    total = len(expected)

    problems = []
    if missing:
        problems.append(f"{missing} missing labels")
    if wrong:
        problems.append(f"{wrong} wrong labels")
    if extra:
        problems.append(f"{extra} extra labels")

    return {
        "ok": not problems,
        "detail": "; ".join(problems),
        "counts": {
            "total": total,
            "correct": correct,
            "missing": missing,
            "wrong_kind": wrong,
            "extra": extra,
        },
    }


def sum_classify(t, n):
    return (
        f"kind accuracy {pct(t['correct'], t['total'])} · "
        f"missing {t['missing']} · wrong {t['wrong_kind']} · "
        f"extra {t['extra']}"
    )


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _judge_candidates(items):
    """Send only id and text to the judge, so it decides on text alone."""
    return [{"id": item["id"], "text": item["text"]} for item in items]


def run_judge(case):
    return llm.judge(
        case["task"],
        _judge_candidates(case["existing"]),
    )


# ---------------------------------------------------------------------------
# Retrieval-only evaluation
# ---------------------------------------------------------------------------

def run_retrieval(case):
    """Retrieve candidates from the isolated evaluation corpus."""
    return retrieval_eval.search_tasks(
        text=case["candidate_task"],
        n_results=3,
    )


def score_retrieval(case, out):
    """Score retrieval relevance, independent of operation selection."""
    expected = set(case["expected_retrieved_tasks"])
    retrieved_ids = [
        item["id"] for item in out
    ]

    hits = sum(task_id in expected for task_id in retrieved_ids)

    recall = hits / len(expected) if expected else None
    hit = int(hits > 0)

    reciprocal_rank = 0.0
    for rank, task_id in enumerate(retrieved_ids, start=1):
        if task_id in expected:
            reciprocal_rank = 1.0 / rank
            break

    # Empty expected lists need a separate no-match/abstention
    # evaluation; top-3 retrieval alone cannot prove no match exists.
    return {
        "ok": bool(hit) if expected else True,
        "detail": (
            f"retrieved={retrieved_ids}, relevant={sorted(expected)}"
        ),
        "counts": {
            "hit_at_3": hit if expected else 0,
            "reciprocal_rank": reciprocal_rank,
            "recall_sum": recall if recall is not None else 0.0,
            "recall_cases": int(bool(expected)),
            "queries_with_relevant": int(bool(expected)),
        },
        "output": out,
    }


def sum_retrieval(totals, n):
    relevant_n = totals["queries_with_relevant"]
    return (
        f"Hit@3 {pct(totals['hit_at_3'], relevant_n)} · "
        f"Recall@3 "
        f"{totals['recall_sum'] / relevant_n:.3f} · "
        f"MRR {totals['reciprocal_rank'] / n:.3f}"
        if relevant_n and n
        else "No relevant retrieval labels to score"
    )



VALID_OPS = {"new", "update", "duplicate", "complete", "cancel", "needs_human"}


def score_judge(c, out):
    if not isinstance(out, dict):
        return {
            "ok": False,
            "detail": f"expected dict output, got {type(out).__name__}",
            "counts": {
                "op_ok": 0,
                "duplicate_tp": 0,
                "duplicate_fp": 0,
                "duplicate_fn": int(c.get("expected_op") == "duplicate"),
                "wrong_task": 0,
            },
        }

    want = c.get("accept", [c["expected_op"]])
    operation = out.get("operation")
    task_id = out.get("task_id")
    expected_op = c["expected_op"]
    expected_duplicate = expected_op == "duplicate"
    predicted_duplicate = operation == "duplicate"

    problems = []
    existing_ids = {
        item["id"] for item in c.get("existing", [])
        if isinstance(item, dict) and "id" in item
    }

    if operation not in VALID_OPS:
        problems.append(f"invalid operation {operation!r}")
    elif operation not in want:
        problems.append(
            f"op {operation}, expected {'/'.join(want)}"
        )

    # Every operation that targets an existing task must use a valid ID.
    if operation in ID_OPS:
        if task_id not in existing_ids:
            problems.append(
                f"task_id {task_id!r} not in existing candidate IDs"
            )
        elif (
            "expected_task_id" in c
            and task_id != c["expected_task_id"]
        ):
            problems.append(
                f"task_id {task_id}, expected {c['expected_task_id']}"
            )

    if operation == "needs_human":
        if not isinstance(out.get("clarifying_question"), str) or not out["clarifying_question"].strip():
            problems.append("needs_human requires a nonempty clarifying_question")

    if operation == "new" and task_id is not None:
        problems.append("new should not target an existing task")

    return {
        "ok": not problems,
        "detail": "; ".join(problems),
        "counts": {
            "op_ok": int(operation in want),
            "duplicate_tp": int(predicted_duplicate and expected_duplicate),
            "duplicate_fp": int(predicted_duplicate and not expected_duplicate),
            "duplicate_fn": int(not predicted_duplicate and expected_duplicate),
            "wrong_task": int(
                operation in ID_OPS
                and task_id in existing_ids
                and "expected_task_id" in c
                and task_id != c["expected_task_id"]
            ),
        },
    }



def sum_judge(t, n):
    tp = t["duplicate_tp"]
    fp = t["duplicate_fp"]
    fn = t["duplicate_fn"]

    precision = pct(tp, tp + fp)
    recall = pct(tp, tp + fn)

    return (
        f"operation accuracy {pct(t['op_ok'], n)} · "
        f"duplicate precision {precision} · "
        f"duplicate recall {recall} · "
        f"wrong-task selections {t['wrong_task']}"
    )



# ---------------------------------------------------------------------------
# Interpret
# ---------------------------------------------------------------------------

# The interpret step drives the agentic InterpretAgent loop (which finishes by
# calling the terminal submit_plan tool) rather than the retired llm.interpret
# one-shot call. run_interpret returns the produced SubmitPlan as a plain dict
# with keys {actions, new_dump, unrelated}; scoring compares it against the
# case's expected routing targets and per-candidate operations.
#
# Design choice (kept runnable + honest): we invoke the real InterpretAgent.run
# over the case's candidates/message. This issues real model calls, like every
# other LLM-backed step in this runner (dump/extract/judge/...), so the eval
# needs the same OpenAI credentials those steps need — it does not fabricate a
# plan from the expected answer (that would make scoring trivially perfect).
# The max_steps fallback (plan is None) is scored as an empty plan so the case
# still produces routing/action numbers instead of raising.


def run_interpret(c):
    agent = InterpretAgent()

    # History-dependent cases carry a `history` array of prior turns
    # ({role, content}). Seed them into a real conversation so the agent's
    # get_history tool returns actual turns; otherwise use the case's conv_id
    # (or None for self-contained cases).
    conv_id = c.get("conv_id")
    if c.get("history"):
        from backend import repositories as repo

        conv_id = repo.get_or_create_conversation()
        for turn in c["history"]:
            repo.add_message(conv_id, turn["role"], turn["content"])

    # Topics the agent may target with change_topic. Carried per-case so a case
    # can control exactly which topics exist (e.g. omit "Gardening" to test
    # that a new topic gets registered). Falls back to the default list.
    topics = c.get("topics", DEFAULT_TOPICS)

    result = agent.run(
        conv_id,
        c["candidates"],
        c["message"],
        topics=topics,
    )
    plan = result.plan
    # Always carry the agent trace + tools_used into the output so the report
    # records agentic behavior (which tools fired, loop steps, stalls,
    # max_steps exhaustion), not just the final plan. The scorer ignores these
    # extra keys. tools_used may be empty / contain "none" markers for
    # no_tool_calls and exhausted steps.
    agent_meta = {
        "tools_used": result.tools_used,
        "trace": result.trace,
    }
    if plan is None:
        # max_steps exhausted with no submit_plan: treat as an empty plan so
        # the case scores routing/action-correctness (likely 0) rather than
        # crashing the runner.
        return {"actions": [], "new_dump": None, "unrelated": None, **agent_meta}
    return {
        "actions": [
            {
                "display_no": a.display_no,
                "operation": a.operation,
                # The operation alone is meaningless for mutating ops —
                # change_text needs the new text, change_kind the new kind,
                # needs_human the question. Carry the full payload so the
                # report shows WHAT the agent changed to (and whether it
                # fabricated or left it empty), not just that it changed.
                "text": a.text,
                "kind": a.kind,
                "topic": a.topic,
                "question": a.question,
            }
            for a in plan.actions
        ],
        "new_dump": plan.new_dump,
        "unrelated": plan.unrelated,
        **agent_meta,
    }


def score_interpret(c, out):
    """Score routing and action-correctness as two independent measures.

    Routing (Req 13.1/13.4): the proportion of content items routed to the
    correct target — either `new_dump`/`unrelated` (leftover/unknown content)
    or `actions` (an existing candidate). We check two expected routing
    signals the case can carry:
      - `expected_other`: whether leftover content should surface via
        new_dump/unrelated (True) or stay empty (False).
      - `expected_action_nos`: the set of candidate display_nos that should be
        touched by an action (i.e. routed to `actions`, not left for routing).

    Action-correctness (Req 13.2/13.5): the proportion of candidates assigned
    the correct operation, keyed by display_no against `expected_operations`.

    When a measure has zero items to evaluate, its counts stay zero and
    sum_interpret reports it as `n/a` (Req 13.6).
    """
    actions = out.get("actions") or []
    by_no = {a["display_no"]: a["operation"] for a in actions}
    action_by_no = {a["display_no"]: a for a in actions}
    wrong = []  # human-readable problems collected across all measures

    # -- Routing measure -------------------------------------------------
    # Content items = the candidates the case expects routed to `actions`
    # plus (optionally) a single "leftover" content item routed to
    # new_dump/unrelated. Each is correct when it lands on its target.
    expected_action_nos = set(c.get("expected_action_nos", []))
    routing_ok = 0
    routing_total = 0

    for no in expected_action_nos:
        routing_total += 1
        if no in by_no:
            routing_ok += 1

    other_present = bool(
        (out.get("new_dump") or "").strip()
        or (out.get("unrelated") or "").strip()
    )
    if "expected_other" in c:
        routing_total += 1
        if other_present == bool(c["expected_other"]):
            routing_ok += 1

    # Precise routing: when the case says what SHOULD land in new_dump, require
    # those keywords (case-insensitive) to appear there. This distinguishes
    # correctly-routed new content from junk routing (e.g. routing a command
    # like "confirm number 7" into new_dump), which the boolean cannot.
    new_dump_keywords = c.get("expected_new_dump_contains")
    if new_dump_keywords:
        routing_total += 1
        blob = (out.get("new_dump") or "").lower()
        if all(k.lower() in blob for k in new_dump_keywords):
            routing_ok += 1
        else:
            missing = [k for k in new_dump_keywords if k.lower() not in blob]
            wrong.append(f"new_dump missing {missing}")

    # -- Action-correctness measure --------------------------------------
    expected_operations = {
        int(no): op
        for no, op in (c.get("expected_operations") or {}).items()
    }
    action_ok = 0
    action_total = 0

    for no, want_op in expected_operations.items():
        action_total += 1
        got_op = by_no.get(no)
        # want_op may be a single op string OR a list of acceptable ops
        # (hard-tier cases where more than one operation is defensible, e.g.
        # "same existing task" → update_task OR reject). Mirrors the accept
        # list used by score_judge.
        accepted = want_op if isinstance(want_op, list) else [want_op]
        if got_op in accepted:
            action_ok += 1
        else:
            wrong.append(f"#{no}: op wanted {'/'.join(accepted)}, got {got_op}")

    # -- Content measure (positive keywords) -----------------------------
    # For text-producing ops the operation alone is not enough — the written
    # text must actually contain the required detail (e.g. the real account
    # number / name), not a fabricated one. `expected_text_contains` maps a
    # display_no to a list of keywords that must ALL appear (case-insensitive
    # substring) in that action's text. A missing action or missing keyword
    # fails the content check for that candidate.
    expected_text = {
        int(no): kws
        for no, kws in (c.get("expected_text_contains") or {}).items()
    }
    content_ok = 0
    content_total = 0

    for no, keywords in expected_text.items():
        content_total += 1
        action = action_by_no.get(no)
        text = (action.get("text") or "").lower() if action else ""
        missing = [k for k in keywords if k.lower() not in text]
        if action and not missing:
            content_ok += 1
        else:
            wrong.append(
                f"#{no}: text missing {missing or 'ACTION'}"
            )

    # -- Topic measure ----------------------------------------------------
    # For change_topic actions, the chosen topic must match (case-insensitive,
    # since the service title-cases). `expected_topic` maps a display_no to the
    # expected topic. Folded into the content measure counts.
    expected_topic = {
        int(no): t
        for no, t in (c.get("expected_topic") or {}).items()
    }
    for no, want_topic in expected_topic.items():
        content_total += 1
        action = action_by_no.get(no)
        got_topic = (action.get("topic") or "") if action else ""
        if action and got_topic.strip().casefold() == want_topic.casefold():
            content_ok += 1
        else:
            wrong.append(
                f"#{no}: topic wanted {want_topic}, got {got_topic or 'ACTION'}"
            )

    # Tool / trace telemetry (summed across cases by sum_interpret). Count each
    # real tool the agent used, how many turns needed >1 model step, and how
    # many stalled (no submit_plan). "none" markers are not real tool calls.
    trace = out.get("trace") or []
    real_tools = [t for t in (out.get("tools_used") or []) if t != "none"]
    steps = {entry.get("step") for entry in trace} or {1}

    problems = []
    if routing_total and routing_ok != routing_total:
        problems.append(f"routing {routing_ok}/{routing_total}")
    if wrong:
        problems.append(" | ".join(wrong))

    # Tool-use expectation: a history-dependent case only passes if the agent
    # actually called the required tool(s). Without this a case could resolve
    # correctly by guessing, never exercising get_history. `expected_tools` is
    # a list of tool names that MUST appear in the trace.
    tool_ok = True
    for required in c.get("expected_tools", []):
        if required not in real_tools:
            tool_ok = False
            problems.append(f"missing tool {required}")

    counts = {
        "routing_ok": routing_ok,
        "routing_total": routing_total,
        "action_ok": action_ok,
        "action_total": action_total,
        "content_ok": content_ok,
        "content_total": content_total,
        "tool_get_history": real_tools.count("get_history"),
        "tool_submit_plan": real_tools.count("submit_plan"),
        "multi_step_turns": int(len(steps) > 1),
        "stalled_turns": int(
            any(e.get("status") in ("no_tool_calls", "exhausted") for e in trace)
        ),
        # History-tool compliance: how many cases that required a tool got it.
        "tool_expect_total": int(bool(c.get("expected_tools"))),
        "tool_expect_ok": int(bool(c.get("expected_tools")) and tool_ok),
    }

    return {
        "ok": not problems,
        "detail": "; ".join(problems),
        "counts": counts,
    }


def sum_interpret(t, n):
    """Two independent measures, each its own line; never a single aggregate.

    Mirrors the two-group reporting style of sum_judge_rag (ROUTING / ACTION).
    A measure with zero items to score reports `n/a` (Req 13.6).
    """
    routing = (
        "ROUTING    " + pct(t["routing_ok"], t["routing_total"])
        if t["routing_total"]
        else "ROUTING    n/a"
    )
    action = (
        "ACTION     " + pct(t["action_ok"], t["action_total"])
        if t["action_total"]
        else "ACTION     n/a"
    )
    content = (
        "CONTENT    " + pct(t["content_ok"], t["content_total"])
        if t.get("content_total")
        else "CONTENT    n/a"
    )
    tools = (
        "TOOLS      "
        f"get_history x{t.get('tool_get_history', 0)}, "
        f"submit_plan x{t.get('tool_submit_plan', 0)} · "
        f"{t.get('multi_step_turns', 0)} multi-step, "
        f"{t.get('stalled_turns', 0)} stalled"
    )
    if t.get("tool_expect_total"):
        tools += (
            " · history-tool "
            + pct(t.get("tool_expect_ok", 0), t["tool_expect_total"])
        )
    return (
        routing + "\n   " + action + "\n   " + content + "\n   " + tools
    )



def run_judge_rag(case):
    """Retrieve candidates, then judge the incoming task.

    JUDGE_RAG_TOP_K and JUDGE_RAG_MIN_SIM override the retrieval top-k and the
    similarity threshold so a sweep can be run without code edits. Defaults
    mirror production (top-k 4, settings.retrieval_min_similarity).
    """
    import os

    top_k = int(os.getenv("JUDGE_RAG_TOP_K", "4"))
    min_sim = float(
        os.getenv("JUDGE_RAG_MIN_SIM", settings.retrieval_min_similarity)
    )

    retrieved = judge_rag_retrieval.search_tasks(
        text=case["task"],
        n_results=top_k,
    )

    retrieved = [
        item for item in retrieved
        if item.get("similarity", 0) >= min_sim
    ]

    existing = [
        {
            "id": item["id"],
            "text": item["text"],
            "score": item.get("similarity"),
        }
        for item in retrieved
    ]

    # The judge decides on text alone; the score is kept in `existing` only
    # for retrieval scoring/reporting, not sent to the model.
    decision = llm.judge(case["task"], _judge_candidates(existing))

    return {
        "decision": decision,
        "retrieved": existing,
    }




def score_judge_rag(case, out):
    """Score the complete retrieval -> judge pipeline."""
    if not isinstance(out, dict) or "decision" not in out:
        return {
            "ok": False,
            "detail": "Invalid judge_rag output",
            "counts": {
                "op_ok": 0,
                "duplicate_tp": 0,
                "duplicate_fp": 0,
                "duplicate_fn": int(
                    case.get("expected_op") == "duplicate"
                ),
                "wrong_task": 0,
                "judge_ok": 0,
                "retrieval_hit": 0,
                "retrieval_labeled": int("expected_task_id" in case),
                "reciprocal_rank": 0.0,
            },
            "output": None,
            "retrieval": {
                "retrieved": [],
                "retrieved_ids": [],
                "expected_task_id": case.get("expected_task_id"),
                "target_found": None,
                "target_rank": None,
                "reciprocal_rank": 0.0,
                "ok": None,
            },
            "inference": {
                "decision": None,
                "ok": False,
                "detail": "Invalid judge_rag output",
            },
        }

    existing = out["retrieved"]
    decision = out["decision"]

    # -- Inference (judge) stage, scored independently of retrieval --------
    # Reuse the judge scorer, validating IDs against the candidates that
    # retrieval actually returned. This measures judge quality on its own,
    # so a retrieval miss does not contaminate the inference metrics.
    judge_case = {**case, "existing": existing}
    scored = score_judge(judge_case, decision)
    judge_ok = scored["ok"]

    # -- Retrieval stage, scored independently of the judge ----------------
    retrieved_ids = [item["id"] for item in existing]
    expected_id = case.get("expected_task_id")
    labeled = expected_id is not None

    target_found = expected_id in retrieved_ids if labeled else None
    # Rank of the expected task in the retrieved list (1-based), 0 if absent.
    target_rank = (
        retrieved_ids.index(expected_id) + 1 if target_found else 0
    )
    reciprocal_rank = (1.0 / target_rank) if target_rank else 0.0
    retrieval_ok = target_found if labeled else None

    # -- Diagnosis: attribute each failure to a stage ----------------------
    problems = []
    if not judge_ok:
        problems.append(f"[judge] {scored['detail']}")
    if retrieval_ok is False:
        problems.append(f"[retrieval] missed expected task {expected_id}")

    # Judge counts stay exactly as score_judge computed them (inference only).
    # Retrieval counts are tracked under a separate namespace.
    counts = dict(scored["counts"])
    counts["judge_ok"] = int(judge_ok)
    counts["retrieval_hit"] = int(target_found is True)
    counts["retrieval_labeled"] = int(labeled)
    counts["reciprocal_rank"] = reciprocal_rank

    return {
        # Overall pass requires both stages to succeed where labeled.
        "ok": judge_ok and (retrieval_ok is not False),
        "detail": "; ".join(problems),
        "counts": counts,
        # Persist both halves of the pipeline so failures can be traced to
        # retrieval vs judgment. `output` is None to avoid duplicating this
        # data in run_case's raw output blob.
        "output": None,
        "retrieval": {
            "retrieved": existing,
            "retrieved_ids": retrieved_ids,
            "expected_task_id": expected_id,
            "target_found": target_found,
            "target_rank": target_rank or None,
            "reciprocal_rank": reciprocal_rank,
            "ok": retrieval_ok,
        },
        "inference": {
            "decision": decision,
            "ok": judge_ok,
            "detail": scored["detail"],
        },
    }


def sum_judge_rag(totals, n):
    """Two independent metric groups: retrieval quality and inference quality."""
    labeled = totals["retrieval_labeled"]

    retrieval = (
        "RETRIEVAL  "
        f"recall@k {pct(totals['retrieval_hit'], labeled)} · "
        f"MRR {totals['reciprocal_rank'] / labeled:.3f}"
        if labeled
        else "RETRIEVAL  no labeled targets to score"
    )

    inference = "INFERENCE  " + sum_judge(totals, n)

    return retrieval + "\n   " + inference

# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEP_FUNCS = {
    "dump": (run_dump, score_dump, sum_dump),
    "extract": (run_extract, score_extract, sum_extract),
    "classify": (run_classify, score_classify, sum_classify),
    "extract_split": (
        run_extract_split,
        score_extract_split,
        sum_extract,
    ),
    "judge": (run_judge, score_judge, sum_judge),
    "interpret": (run_interpret, score_interpret, sum_interpret),
    "retrieval": (
    run_retrieval,
    score_retrieval,
    sum_retrieval,
    ),
    "judge_rag": (run_judge_rag, score_judge_rag, sum_judge_rag),
}


# ---------------------------------------------------------------------------
# Case loading and execution
# ---------------------------------------------------------------------------

def setup_judge_rag_retrieval(tasks):
    # This is safe because this service uses a dedicated judge_rag collection.
    judge_rag_retrieval.collection.delete(
        where={"record_type": "task"}
    )

    records = [
        (
            str(task["id"]),
            task["text"],
            {
                "task_id": str(task["id"]),
                "topic": task.get("topic", "INBOX"),
                "status": task.get("status", "open"),
                "record_type": "task",
            },
        )
        for task in tasks
    ]

    judge_rag_retrieval.upsert_batch(records)


def load(step, cases_name=None):
    name = cases_name or CASE_FILES.get(step, step)
    path = find_case_file(name)
    data = json.loads(path.read_text(encoding="utf-8"))

    if step == "judge_rag":
        tasks = data["existing_tasks"]
        setup_judge_rag_retrieval(tasks)

        return [
            {**case, "existing_tasks": tasks}
            for case in data["cases"]
        ]

    cases = data

    # Scaling cases can reference inputs and sizes from dump.json.
    if any("dump_id" in case for case in cases):
        dump_cases = json.loads(
            find_case_file("dump").read_text(encoding="utf-8")
        )
        dump_by_id = {case["id"]: case for case in dump_cases}

        for case in cases:
            if "dump_id" in case:
                source = dump_by_id[case["dump_id"]]
                case.setdefault("input", source["input"])
                case.setdefault("size", source["size"])

    return cases


def run_case(step, case):
    run, score, _ = STEP_FUNCS[step]
    llm.start_tracking()
    start = time.perf_counter()

    try:
        output = run(case)
        result = {
            "id": case["id"],
            "output": output,
            **score(case, output),
        }
    except Exception as exc:
        result = {
            "id": case["id"],
            "ok": False,
            "detail": f"ERROR {type(exc).__name__}: {exc}",
            "counts": {},
            "output": None,
        }

    calls = llm.pop_calls()
    parts = [p for p in map(call_cost_parts, calls) if p is not None]
    priced = [sum(column) for column in zip(*parts)] if parts else [0, 0, 0]

    result["usage"] = {
        "calls": len(calls),
        "latency_s": time.perf_counter() - start,
        "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
        "completion_tokens": sum(call["completion_tokens"] for call in calls),
        "cached_tokens": sum(call["cached_tokens"] for call in calls),
        "cost_input_usd": priced and priced[0],
        "cost_cached_input_usd": priced and priced[1],
        "cost_output_usd": priced and priced[2],
        "cost_usd": priced and sum(priced),
        "models": sorted({call["model"] for call in calls}),
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    # The report uses unicode (· ✗ etc.). On Windows the default console
    # codec is cp1252, which raises UnicodeEncodeError mid-run and can abort
    # before the --json report is written. Force UTF-8 so output is portable.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=list(STEP_FUNCS),
        action="append",
        help="repeatable; default: all steps",
    )
    parser.add_argument(
        "--cases",
        help="case file name without .json, e.g. extract or interpret_agent_hard",
    )
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", help="write full results to this path")
    parser.add_argument(
        "--fail-under",
        type=float,
        help="exit 1 if any selected step's case accuracy is below this",
    )
    parser.add_argument("--price-in", type=float)
    parser.add_argument("--price-out", type=float)
    parser.add_argument("--price-cached", type=float)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.price_in is not None and args.price_out is not None:
        llm_pricing.OVERRIDE = (
            args.price_in,
            args.price_cached,
            args.price_out,
        )

    if args.model:
        llm._llm = llm.LLMClient(args.model)

    steps = args.step or list(STEP_FUNCS)
    model = args.model or llm.DEFAULT_MODEL
    report = {}
    worst = 1.0
    start = time.perf_counter()

    print(f"model: {model}")

    for step in steps:
        cases = load(step, args.cases)

        with ThreadPoolExecutor(args.workers) as executor:
            results = list(
                executor.map(lambda case: run_case(step, case), cases)
            )

        totals = Counter()
        for result in results:
            totals.update(result["counts"])

        passed = sum(result["ok"] for result in results)
        accuracy = passed / len(results) if results else 1.0
        worst = min(worst, accuracy)

        prompts = ", ".join(
            llm.get_prompt_for_operation(name)
            for name in LLM_STEPS[step]
        )

        print(f"\n== {step}: {pct(passed, len(results))} cases passed [{prompts}]")
        print("   " + STEP_FUNCS[step][2](totals, len(results)))
        print("   " + usage_line(results))

        groups = list(
            dict.fromkeys(
                case["size"] for case in cases if case.get("size")
            )
        )

        for group in groups:
            indices = [
                i for i, case in enumerate(cases)
                if case.get("size") == group
            ]
            group_totals = Counter()
            for i in indices:
                group_totals.update(results[i]["counts"])

            group_usage = [results[i]["usage"] for i in indices]
            group_costs = cost_split(group_usage)

            cost_text = ""
            if any(usage["calls"] for usage in group_usage):
                average_latency = (
                    sum(u["latency_s"] for u in group_usage)
                    / len(group_usage)
                )
                if group_costs:
                    cost_text = (
                        f" · {average_latency:.1f}s "
                        f"{money(group_costs[0] / len(group_usage))}/case "
                        f"({money(group_costs[1] / len(group_usage))} in + "
                        f"{money(group_costs[2] / len(group_usage))} out)"
                    )
                else:
                    cost_text = f" · {average_latency:.1f}s cost n/a"

            group_passed = sum(results[i]["ok"] for i in indices)
            summary = STEP_FUNCS[step][2](group_totals, len(indices))
            print(
                f"   {group:<14} {group_passed}/{len(indices)} pass · "
                f"{summary}{cost_text}"
            )

        for result in results:
            if not result["ok"]:
                print(f"   ✗ {result['id']}: {result['detail']}")
                if args.verbose:
                    print(
                        "       output: "
                        + json.dumps(result["output"], ensure_ascii=False)
                    )

        priced = [
            result["usage"]["cost_usd"]
            for result in results
            if result["usage"]["cost_usd"] is not None
        ]
        costs = cost_split([result["usage"] for result in results])

        report[step] = {
            "accuracy": accuracy,
            "passed": passed,
            "total": len(results),
            "prompts": prompts,
            "latency_p50_s": percentile(
                [r["usage"]["latency_s"] for r in results], 50
            ),
            "latency_p95_s": percentile(
                [r["usage"]["latency_s"] for r in results], 95
            ),
            "cost_usd": sum(priced) if priced else None,
            "cost_input_usd": costs and costs[1],
            "cost_output_usd": costs and costs[2],
            "prompt_tokens": sum(
                r["usage"]["prompt_tokens"] for r in results
            ),
            "completion_tokens": sum(
                r["usage"]["completion_tokens"] for r in results
            ),
            "cases": results,
        }

    all_usage = [
        case["usage"]
        for step_result in report.values()
        for case in step_result["cases"]
    ]
    unpriced = sorted({
        model_name
        for usage in all_usage
        if usage["calls"] and usage["cost_usd"] is None
        for model_name in usage["models"]
    })

    if any(usage["calls"] for usage in all_usage):
        total_costs = cost_split(all_usage)
        models = sorted({
            model_name
            for usage in all_usage
            for model_name in usage["models"]
        })

        total_cost_text = (
            f"{money(total_costs[0])} "
            f"({money(total_costs[1])} input + "
            f"{money(total_costs[2])} output)"
            if total_costs
            else "n/a"
        )

        print(
            f"\nrun total: {total_cost_text} · "
            f"{sum(u['calls'] for u in all_usage)} calls · "
            f"{time.perf_counter() - start:.0f}s wall · "
            f"served by {', '.join(models)}"
        )

    if unpriced:
        print(
            f"note: no price for {unpriced}; cost excludes those calls. "
            "Add it to PRICES or pass --price-in/--price-out "
            "(USD per 1M tokens)."
        )

    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {"model": model, "steps": report},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nresults written to {args.json}")

    if args.fail_under is not None and worst < args.fail_under:
        print(
            f"\nFAIL: lowest step accuracy {worst:.0%} "
            f"< {args.fail_under:.0%}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
