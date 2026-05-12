"""Tests for :mod:`hallucination_guard.nodes.factcheck_gate`."""

from __future__ import annotations

import pytest

from hallucination_guard.domain.general import GeneralDomain
from hallucination_guard.exceptions import GraphError
from hallucination_guard.nodes.factcheck_gate import FactCheckGate
from hallucination_guard.schemas import Claim, GroundedOutput
from hallucination_guard.state import FailReason, GraphState


@pytest.fixture
def gate() -> FactCheckGate:
    return FactCheckGate(GeneralDomain())


def _state_with(*claims: Claim) -> GraphState:
    return GraphState(
        user_query="q",
        research_output=GroundedOutput(claims=list(claims)),
    )


def test_passes_with_high_confidence_and_valid_sources(gate: FactCheckGate) -> None:
    state = _state_with(
        Claim(text="claim A", confidence=0.9, sources=["https://example.com/1"]),
    )
    result = gate(state)
    assert result.gate_result == "PASS"
    assert result.fail_reason is None
    assert result.fail_history == []


def test_fails_on_low_confidence(gate: FactCheckGate) -> None:
    state = _state_with(
        Claim(text="risky", confidence=0.3, sources=["https://example.com/1"]),
    )
    result = gate(state)
    assert result.gate_result == "FAIL"
    assert result.fail_reason == FailReason.LOW_CONFIDENCE
    assert result.fail_history == ["low_confidence:risky"]


def test_fails_on_empty_sources(gate: FactCheckGate) -> None:
    state = _state_with(
        Claim(text="no-src", confidence=0.9, sources=[]),
    )
    result = gate(state)
    assert result.gate_result == "FAIL"
    assert result.fail_reason == FailReason.NO_SOURCE
    assert result.fail_history == ["no_source:no-src"]


def test_fails_on_only_invalid_sources(gate: FactCheckGate) -> None:
    state = _state_with(
        Claim(text="bad-src", confidence=0.9, sources=["http://insecure.example"]),
    )
    result = gate(state)
    assert result.gate_result == "FAIL"
    assert result.fail_reason == FailReason.NO_SOURCE


def test_passes_when_at_least_one_source_is_valid(gate: FactCheckGate) -> None:
    state = _state_with(
        Claim(
            text="mixed-src",
            confidence=0.9,
            sources=["http://insecure.example", "https://ok.example"],
        ),
    )
    result = gate(state)
    assert result.gate_result == "PASS"


def test_confidence_check_takes_precedence_over_source_check(
    gate: FactCheckGate,
) -> None:
    state = _state_with(
        Claim(text="x", confidence=0.1, sources=[]),
    )
    result = gate(state)
    assert result.fail_reason == FailReason.LOW_CONFIDENCE


def test_appends_one_history_entry_per_failing_claim(gate: FactCheckGate) -> None:
    state = _state_with(
        Claim(text="A", confidence=0.1, sources=["https://example.com/1"]),
        Claim(text="B", confidence=0.2, sources=["https://example.com/2"]),
        Claim(text="C", confidence=0.9, sources=["https://example.com/3"]),
    )
    result = gate(state)
    assert result.fail_history == ["low_confidence:A", "low_confidence:B"]


def test_preserves_existing_fail_history(gate: FactCheckGate) -> None:
    state = GraphState(
        user_query="q",
        research_output=GroundedOutput(
            claims=[Claim(text="x", confidence=0.1, sources=["https://e.com"])]
        ),
        fail_history=["critic_rejected:earlier"],
    )
    result = gate(state)
    assert result.fail_history == [
        "critic_rejected:earlier",
        "low_confidence:x",
    ]


def test_raises_when_research_output_is_missing(gate: FactCheckGate) -> None:
    state = GraphState(user_query="q", research_output=None)
    with pytest.raises(GraphError):
        gate(state)


def test_does_not_mutate_input_state(gate: FactCheckGate) -> None:
    state = _state_with(
        Claim(text="x", confidence=0.1, sources=["https://example.com"]),
    )
    original_history = list(state.fail_history)
    gate(state)
    assert state.fail_history == original_history
    assert state.gate_result is None
