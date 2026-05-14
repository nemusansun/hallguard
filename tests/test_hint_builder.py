"""Tests for :mod:`hallucination_guard.retry.hint_builder`."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from hallucination_guard.domain.base import DomainConfig
from hallucination_guard.retry.directive import RetryDirective
from hallucination_guard.retry.hint_builder import RetryHintBuilder
from hallucination_guard.schemas import GroundedOutput
from hallucination_guard.state import FailReason, GraphState


class _StubDomain(DomainConfig):
    """Records ``retry_instruction`` calls so we can assert the builder
    delegates wording entirely to the domain — no fallback or hardcoded map."""

    def __init__(self, mapping: dict[FailReason, str]) -> None:
        self._mapping = mapping
        self.calls: list[FailReason] = []

    @property
    def confidence_threshold(self) -> float:
        return 0.0

    def is_valid_source(self, url: str) -> bool:
        return True

    def critic_prompt(self) -> str:
        return ""

    def output_schema(self) -> type[BaseModel]:
        return GroundedOutput

    def system_prompt(self) -> str:
        return ""

    def format_retry_directive(
        self, base_prompt: str, directive: RetryDirective
    ) -> str:
        return base_prompt

    def retry_instruction(self, fail_reason: FailReason) -> str:
        self.calls.append(fail_reason)
        return self._mapping[fail_reason]


def _stub() -> _StubDomain:
    return _StubDomain({r: f"phrase-{r.value}" for r in FailReason})


@pytest.mark.parametrize("reason", list(FailReason))
def test_build_returns_domain_supplied_instruction(reason: FailReason) -> None:
    """The builder forwards wording from ``domain.retry_instruction``."""
    domain = _stub()
    state = GraphState(user_query="q", fail_reason=reason)

    directive = RetryHintBuilder.build(state, domain)

    assert isinstance(directive, RetryDirective)
    assert directive.fix_instruction == f"phrase-{reason.value}"
    assert domain.calls == [reason]


def test_build_raises_when_fail_reason_is_none() -> None:
    """No fail_reason ⇒ no directive to build."""
    state = GraphState(user_query="q", fail_reason=None)

    with pytest.raises(ValueError):
        RetryHintBuilder.build(state, _stub())


def test_build_does_not_leak_fail_history_strings() -> None:
    """Raw ``fail_history`` text must not appear inside the directive.

    The fix_instruction comes only from ``domain.retry_instruction``, and
    ``forbidden_claims`` contains the stripped claim text — never the
    ``"<reason>:<claim>"`` raw entry.
    """
    raw_history = [
        "low_confidence:internal note that must not leak",
        "critic_rejected:claim ALPHA",
        "no_source:another internal note",
    ]
    state = GraphState(
        user_query="q",
        fail_reason=FailReason.LOW_CONFIDENCE,
        fail_history=raw_history,
    )

    directive = RetryHintBuilder.build(state, _stub())

    assert directive.fix_instruction == "phrase-low_confidence"
    for raw in raw_history:
        assert raw not in directive.fix_instruction
        assert raw not in directive.forbidden_claims


def test_build_populates_forbidden_claims_on_critic_rejected() -> None:
    """``CRITIC_REJECTED`` entries flow into ``forbidden_claims`` (stripped)."""
    state = GraphState(
        user_query="q",
        fail_reason=FailReason.CRITIC_REJECTED,
        fail_history=[
            "critic_rejected:claim ALPHA",
            "low_confidence:irrelevant",
            "critic_rejected: claim BETA ",
        ],
    )

    directive = RetryHintBuilder.build(state, _stub())

    assert directive.fix_instruction == "phrase-critic_rejected"
    assert directive.forbidden_claims == ["claim ALPHA", "claim BETA"]


def test_retry_directive_is_frozen() -> None:
    """The directive is immutable once constructed."""
    directive = RetryDirective(fix_instruction="x", forbidden_claims=["a"])

    with pytest.raises(Exception):
        directive.fix_instruction = "y"  # type: ignore[misc]
