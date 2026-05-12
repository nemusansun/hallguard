"""Tests for :mod:`hallucination_guard.nodes.retry_node`."""

from __future__ import annotations

from hallucination_guard.nodes.retry_node import RetryNode
from hallucination_guard.state import FailReason, GraphState


def test_retry_count_increments() -> None:
    state = GraphState(user_query="q", retry_count=0)
    result = RetryNode()(state)
    assert result.retry_count == 1


def test_does_not_mutate_input_state() -> None:
    state = GraphState(user_query="q", retry_count=1)
    RetryNode()(state)
    assert state.retry_count == 1


def test_clears_stale_gate_and_critic_results() -> None:
    state = GraphState(
        user_query="q",
        gate_result="FAIL",
        critic_result="FAIL",
    )
    result = RetryNode()(state)
    assert result.gate_result is None
    assert result.critic_result is None


def test_preserves_fail_reason_and_history_for_hint_builder() -> None:
    state = GraphState(
        user_query="q",
        fail_reason=FailReason.LOW_CONFIDENCE,
        fail_history=["low_confidence:foo"],
    )
    result = RetryNode()(state)
    assert result.fail_reason == FailReason.LOW_CONFIDENCE
    assert result.fail_history == ["low_confidence:foo"]


def test_multiple_invocations_compound_retry_count() -> None:
    node = RetryNode()
    state = GraphState(user_query="q", retry_count=0)
    state = node(state)
    state = node(state)
    state = node(state)
    assert state.retry_count == 3
