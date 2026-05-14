"""Tests for the reducer-aware ``_wrap`` / ``_merge_update`` infrastructure."""

from __future__ import annotations

from hallucination_guard.graph import (
    _ADDITIVE_FIELDS,
    _build_update_dict,
    _coerce_state,
)
from hallucination_guard.state import GraphState


def test_additive_fields_detected() -> None:
    """``_ADDITIVE_FIELDS`` must contain the two reducer-annotated list fields."""
    assert "fail_history" in _ADDITIVE_FIELDS
    assert "branch_outputs" in _ADDITIVE_FIELDS


def test_additive_fields_excludes_non_reducer() -> None:
    """Non-reducer fields must NOT appear in ``_ADDITIVE_FIELDS``."""
    for name in ("user_query", "research_output", "retry_count", "is_success"):
        assert name not in _ADDITIVE_FIELDS


def test_build_update_dict_returns_delta_for_fail_history() -> None:
    """``_build_update_dict`` returns only newly appended items for reducer fields."""
    old = GraphState(user_query="q", fail_history=["a", "b"])
    new = old.with_update(fail_history=["a", "b", "c", "d"])
    result = _build_update_dict(old, new)
    assert result["fail_history"] == ["c", "d"]


def test_build_update_dict_returns_delta_for_branch_outputs() -> None:
    old = GraphState(user_query="q", branch_outputs=["x"])
    new = old.with_update(branch_outputs=["x", "y"])
    result = _build_update_dict(old, new)
    assert result["branch_outputs"] == ["y"]


def test_build_update_dict_returns_full_value_for_non_reducer() -> None:
    """Non-reducer fields appear only when their value changed."""
    old = GraphState(user_query="q", retry_count=0)
    new = old.with_update(retry_count=2)
    result = _build_update_dict(old, new)
    assert result["retry_count"] == 2
    # Unchanged fields are omitted.
    assert "user_query" not in result


def test_build_update_dict_empty_when_no_change() -> None:
    old = GraphState(user_query="q", fail_history=["a"])
    new = old.with_update()  # no change
    result = _build_update_dict(old, new)
    # Empty deltas and unchanged fields are omitted entirely.
    assert "fail_history" not in result
    assert "branch_outputs" not in result


def test_merge_update_applies_additive_reducer() -> None:
    """``_merge_update`` must concatenate deltas for reducer fields."""
    from hallucination_guard.graph import Graph

    accumulated = GraphState(user_query="q", fail_history=["old"])
    update = {"fail_history": ["new1", "new2"]}  # delta from _wrap
    field_names = set(GraphState.model_fields)

    merged = Graph._merge_update(accumulated, update, field_names)
    assert merged is not None
    assert merged.fail_history == ["old", "new1", "new2"]


def test_merge_update_applies_additive_for_branch_outputs() -> None:
    from hallucination_guard.graph import Graph

    accumulated = GraphState(user_query="q", branch_outputs=["x"])
    update = {"branch_outputs": ["y"]}
    field_names = set(GraphState.model_fields)

    merged = Graph._merge_update(accumulated, update, field_names)
    assert merged is not None
    assert merged.branch_outputs == ["x", "y"]
