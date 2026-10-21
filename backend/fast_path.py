"""Deterministic fast path for resolving pending candidates.

Parses simple, unambiguous replies ("confirm all", "reject #2", "yes to 1 and 3")
without calling the LLM. Anything that is not a clean match returns None so the
caller falls through to the LLM router/interpreter.

Precision over recall — match only unambiguous intent, else return None so the
LLM handles it. In particular, defer to the LLM when a target has an open
question (resolving it deterministically is the risky case) or references an
unknown number (so the LLM can explain rather than silently no-op). Bulk
confirm/reject skips questioned candidates, reporting them separately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# "yes", "ok", "confirm", "confirm all", "accept all", "looks good", "all good"
_BULK_CONFIRM = re.compile(
    r"^\s*(yes|yep|yeah|y|ok|okay|confirm|confirm all|accept|accept all|"
    r"looks good|all good|sounds good|save all)\s*[.!]*\s*$",
    re.I,
)

# "no", "reject", "reject all", "discard all", "none", "nothing"
_BULK_REJECT = re.compile(
    r"^\s*(no|nope|n|reject|reject all|discard|discard all|none|nothing|"
    r"delete all)\s*[.!]*\s*$",
    re.I,
)

# "confirm 2", "confirm #2", "yes to 1 and 3", "accept 1, 2, 4"
_TARGET_CONFIRM = re.compile(
    r"^\s*(?:yes\s+to|confirm|accept|save|keep)\s+(?P<nums>[\d,\s#and]+?)\s*[.!]*\s*$",
    re.I,
)

# "reject 2", "reject #2 and #4", "no to 3", "discard 1, 2"
_TARGET_REJECT = re.compile(
    r"^\s*(?:no\s+to|reject|discard|delete|remove|drop)\s+(?P<nums>[\d,\s#and]+?)\s*[.!]*\s*$",
    re.I,
)

_NUM = re.compile(r"\d+")


@dataclass
class FastPathPlan:
    """A resolved deterministic plan.

    action is "confirm" or "reject". targets are the display_nos to act on.
    skipped_questioned are display_nos excluded from a bulk action because they
    have an open question.
    """
    action: str
    targets: list[int]
    skipped_questioned: list[int] = field(default_factory=list)


def _parse_numbers(blob: str) -> list[int]:
    """Extract display numbers from a reference blob like "1, 3 and #4"."""
    return [int(match.group()) for match in _NUM.finditer(blob)]


def parse(text: str, candidates: list[dict]) -> FastPathPlan | None:
    """Return a deterministic plan, or None to defer to the LLM.

    candidates is the list of currently pending candidate dicts (each having
    at least "display_no" and "question").
    """
    if not text or not candidates:
        return None

    by_no = {c["display_no"]: c for c in candidates}
    free = [c["display_no"] for c in candidates if not c.get("question")]

    # --- Bulk commands ---------------------------------------------------
    if _BULK_CONFIRM.match(text):
        questioned = [c["display_no"] for c in candidates if c.get("question")]
        if not free:
            # Everything is blocked by a question; nothing to confirm in bulk.
            return None
        return FastPathPlan("confirm", sorted(free), sorted(questioned))

    if _BULK_REJECT.match(text):
        # Skip questioned items here too, to stay symmetric with bulk confirm.
        questioned = [c["display_no"] for c in candidates if c.get("question")]
        if not free:
            return None
        return FastPathPlan("reject", sorted(free), sorted(questioned))

    # --- Targeted commands ----------------------------------------------
    for pattern, action in (
        (_TARGET_CONFIRM, "confirm"),
        (_TARGET_REJECT, "reject"),
    ):
        match = pattern.match(text)
        if not match:
            continue

        numbers = _parse_numbers(match.group("nums"))
        if not numbers:
            return None

        # Every referenced number must be a currently pending candidate.
        if any(n not in by_no for n in numbers):
            return None

        # Defer to the LLM if any target has an open question (risky to
        # resolve deterministically).
        if any(by_no[n].get("question") for n in numbers):
            return None

        # Dedupe while preserving a stable, sorted order.
        targets = sorted(set(numbers))
        return FastPathPlan(action, targets)

    # Not a clean deterministic command.
    return None
