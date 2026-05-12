"""Tests for :mod:`hallucination_guard.nodes.error_output`."""

from __future__ import annotations

from hallucination_guard.nodes.error_output import ErrorOutput
from hallucination_guard.state import FailReason, GraphState


def test_marks_run_as_unsuccessful() -> None:
    state = GraphState(user_query="q", retry_count=3, max_retries=3)
    result = ErrorOutput()(state)
    assert result.is_success is False
    assert result.final_output is None


def test_error_message_mentions_max_retries_value() -> None:
    state = GraphState(user_query="q", max_retries=5, retry_count=5)
    result = ErrorOutput()(state)
    assert result.error_message is not None
    assert "max_retries (5)" in result.error_message


def test_error_message_includes_last_fail_reason_and_count() -> None:
    state = GraphState(
        user_query="q",
        max_retries=3,
        retry_count=3,
        fail_reason=FailReason.CRITIC_REJECTED,
        fail_history=["critic_rejected:foo", "critic_rejected:bar"],
    )
    result = ErrorOutput()(state)
    assert result.error_message is not None
    assert "critic_rejected" in result.error_message
    assert "total_failures=2" in result.error_message


def test_error_message_falls_back_to_unknown_when_no_fail_reason() -> None:
    state = GraphState(user_query="q", retry_count=3, max_retries=3)
    result = ErrorOutput()(state)
    assert result.error_message is not None
    assert "unknown" in result.error_message


def test_does_not_mutate_input_state() -> None:
    state = GraphState(user_query="q", retry_count=3, max_retries=3)
    ErrorOutput()(state)
    assert state.is_success is False  # default
    assert state.error_message is None
