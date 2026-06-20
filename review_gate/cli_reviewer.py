"""System 6 — CLI approve/reject reviewer for the live demo.

A minimal terminal reviewer that walks the pending queue, prints each item's full
context (content, agent, risk, fired signals, taint trace), and records the
human's decision. `auto_review` is the non-interactive counterpart used by tests
and the scripted demo.
"""

from __future__ import annotations

from typing import Callable, List

from .review_queue import APPROVE, REJECT, ReviewItem, ReviewQueue


def _render(item: ReviewItem) -> str:
    lines = [
        "─" * 60,
        f"REVIEW #{item.id}  | agent={item.agent}  risk={item.risk_score:.2f}",
        "─" * 60,
        f"content : {item.content[:300]}",
        f"signals : {', '.join(item.signals) or '(none)'}",
        f"taint   : {' -> '.join(item.taint_trace) or '(none)'}",
    ]
    if item.ground_truth:
        lines.append(f"(ground truth: {item.ground_truth})")
    return "\n".join(lines)


def review_cli(queue: ReviewQueue) -> List[ReviewItem]:
    """Interactive reviewer. Reads a/r/s from stdin per pending item."""
    decided: List[ReviewItem] = []
    for item in queue.pending():
        print(_render(item))
        choice = input("  [a]pprove / [r]eject / [s]kip ? ").strip().lower()
        if choice.startswith("a"):
            decided.append(queue.decide(item.id, APPROVE, note="cli approve"))
        elif choice.startswith("r"):
            decided.append(queue.decide(item.id, REJECT, note="cli reject"))
        else:
            print("  skipped.")
    return decided


def auto_review(
    queue: ReviewQueue, decision_fn: Callable[[ReviewItem], str]
) -> List[ReviewItem]:
    """Non-interactive reviewer driven by a decision function (item -> 'approve'/'reject')."""
    decided: List[ReviewItem] = []
    for item in list(queue.pending()):
        decided.append(queue.decide(item.id, decision_fn(item)))
    return decided
