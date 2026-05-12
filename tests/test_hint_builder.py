"""Tests for :mod:`hallucination_guard.retry.hint_builder`."""

from __future__ import annotations

import pytest

from hallucination_guard.domain.base import Locale
from hallucination_guard.retry.directive import RetryDirective
from hallucination_guard.retry.hint_builder import RetryHintBuilder
from hallucination_guard.state import FailReason, GraphState


@pytest.mark.parametrize("locale", ["en", "ja"])
@pytest.mark.parametrize("reason", list(FailReason))
def test_build_returns_mapped_instruction_for_each_reason(
    reason: FailReason, locale: Locale
) -> None:
    """Every ``(locale, FailReason)`` pair resolves to its entry in ``INSTRUCTION_MAPS``."""
    state = GraphState(user_query="q", fail_reason=reason)
    directive = RetryHintBuilder.build(state, locale=locale)

    assert isinstance(directive, RetryDirective)
    assert directive.fix_instruction == RetryHintBuilder.INSTRUCTION_MAPS[locale][reason]


def test_build_defaults_to_english_when_locale_is_omitted() -> None:
    """The default locale is ``"en"`` so callers without a locale get English."""
    state = GraphState(user_query="q", fail_reason=FailReason.LOW_CONFIDENCE)
    directive = RetryHintBuilder.build(state)

    assert (
        directive.fix_instruction
        == RetryHintBuilder.INSTRUCTION_MAPS["en"][FailReason.LOW_CONFIDENCE]
    )


def test_build_raises_when_fail_reason_is_none() -> None:
    """No fail_reason ⇒ no directive to build."""
    state = GraphState(user_query="q", fail_reason=None)

    with pytest.raises(ValueError):
        RetryHintBuilder.build(state)


def test_build_does_not_leak_fail_history_strings() -> None:
    """Raw ``fail_history`` text must not appear inside the directive.

    The fix_instruction comes only from ``INSTRUCTION_MAPS``, and
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

    directive = RetryHintBuilder.build(state)

    assert (
        directive.fix_instruction
        == RetryHintBuilder.INSTRUCTION_MAPS["en"][FailReason.LOW_CONFIDENCE]
    )
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

    directive = RetryHintBuilder.build(state)

    assert (
        directive.fix_instruction
        == RetryHintBuilder.INSTRUCTION_MAPS["en"][FailReason.CRITIC_REJECTED]
    )
    assert directive.forbidden_claims == ["claim ALPHA", "claim BETA"]


def test_japanese_and_english_fix_instructions_differ() -> None:
    """Each (locale, reason) pair gets a distinct phrase from the maps.

    Guards against accidentally collapsing the two locale maps into a single
    language and silently regressing localized prompts.
    """
    for reason in FailReason:
        en = RetryHintBuilder.INSTRUCTION_MAPS["en"][reason]
        ja = RetryHintBuilder.INSTRUCTION_MAPS["ja"][reason]
        assert en != ja


def test_retry_directive_is_frozen() -> None:
    """The directive is immutable once constructed."""
    directive = RetryDirective(fix_instruction="x", forbidden_claims=["a"])

    with pytest.raises(Exception):
        directive.fix_instruction = "y"  # type: ignore[misc]
