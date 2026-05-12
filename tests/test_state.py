"""Tests for :mod:`hallucination_guard.state`."""

from __future__ import annotations

from hallucination_guard.state import FailReason, GraphState


def test_with_update_returns_new_instance() -> None:
    """``with_update`` must not mutate the source state."""
    original = GraphState(user_query="q")
    updated = original.with_update(retry_count=1)

    assert updated is not original
    assert updated.retry_count == 1
    assert original.retry_count == 0


def test_with_update_preserves_unchanged_fields() -> None:
    """Fields not passed to ``with_update`` carry over unchanged."""
    original = GraphState(user_query="q", max_retries=5)
    updated = original.with_update(retry_count=2)

    assert updated.user_query == "q"
    assert updated.max_retries == 5
    assert updated.retry_count == 2


def test_with_update_increment_does_not_mutate_original() -> None:
    """Spec: incrementing retry_count via with_update leaves the original at 0."""
    original = GraphState(user_query="q")
    updated = original.with_update(retry_count=original.retry_count + 1)

    assert original.retry_count == 0
    assert updated.retry_count == 1


def test_fail_history_default_is_independent_list() -> None:
    """Two fresh states must not share the same ``fail_history`` list object."""
    s1 = GraphState(user_query="q1")
    s2 = GraphState(user_query="q2")

    s1.fail_history.append("low_confidence:test")
    assert s2.fail_history == []


def test_get_rejected_claims_filters_critic_rejected_only() -> None:
    """Only entries prefixed with ``critic_rejected:`` are returned."""
    state = GraphState(
        user_query="q",
        fail_history=[
            "low_confidence:claim A",
            "critic_rejected:claim B",
            "no_source:claim C",
            "critic_rejected: claim D ",
        ],
    )

    assert state.get_rejected_claims() == ["claim B", "claim D"]


def test_get_rejected_claims_empty_when_no_critic_rejections() -> None:
    state = GraphState(
        user_query="q",
        fail_history=["low_confidence:x", "no_source:y"],
    )
    assert state.get_rejected_claims() == []


def test_fail_reason_is_string_enum() -> None:
    """``FailReason`` inherits ``str`` so values can be compared as strings."""
    assert FailReason.LOW_CONFIDENCE == "low_confidence"
    assert FailReason.NO_SOURCE == "no_source"
    assert FailReason.CRITIC_REJECTED == "critic_rejected"
