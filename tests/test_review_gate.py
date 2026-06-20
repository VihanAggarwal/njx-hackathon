"""Tests for System 6 — the human review gate."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from review_gate import (
    APPROVE,
    AUTO_ALLOW,
    AUTO_BLOCK,
    REJECT,
    REVIEW,
    ReviewQueue,
    auto_review,
)

CFG = {"thresholds": {"review_gate_block": 0.8, "review_gate_review": 0.3}}


def test_routing_thresholds():
    q = ReviewQueue(CFG)
    assert q.route(0.1) == AUTO_ALLOW
    assert q.route(0.3) == REVIEW
    assert q.route(0.5) == REVIEW
    assert q.route(0.8) == REVIEW
    assert q.route(0.81) == AUTO_BLOCK


def test_submit_routes_and_queues():
    q = ReviewQueue(CFG)
    a = q.submit("hello", "agentA", risk_score=0.1)
    b = q.submit("maybe bad", "agentA", risk_score=0.5)
    c = q.submit("clearly bad", "agentA", risk_score=0.95)
    assert a.routing == AUTO_ALLOW and q.final_verdict(a) == "allow"
    assert c.routing == AUTO_BLOCK and q.final_verdict(c) == "block"
    assert b.routing == REVIEW
    assert [i.id for i in q.pending()] == [b.id]  # only the mid-risk one queued


def test_simulated_human_uses_ground_truth():
    q = ReviewQueue(CFG)
    q.submit("injection", "a", 0.5, ground_truth="attack")
    q.submit("legit", "a", 0.4, ground_truth="benign")
    q.drain_simulated()
    assert len(q.rejections()) == 1
    assert len(q.approvals()) == 1
    assert q.pending() == []


def test_final_verdict_after_human_decision():
    q = ReviewQueue(CFG)
    item = q.submit("review me", "a", 0.5, ground_truth="attack")
    assert q.final_verdict(item) == "pending"
    q.simulate_decision(item)
    assert q.final_verdict(item) == "block"


def test_reject_callback_fires():
    q = ReviewQueue(CFG)
    captured = []
    q.on_reject = lambda it: captured.append(it)
    item = q.submit("bad", "a", 0.5)
    q.decide(item.id, REJECT, note="manual")
    assert captured and captured[0].id == item.id


def test_auto_review_helper():
    q = ReviewQueue(CFG)
    q.submit("x", "a", 0.5)
    q.submit("y", "a", 0.6)
    auto_review(q, lambda it: REJECT)
    assert len(q.rejections()) == 2
