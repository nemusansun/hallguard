"""Tests for :mod:`hallucination_guard.domain.general`."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from hallucination_guard.domain.general import GeneralDomain
from hallucination_guard.retry.directive import RetryDirective
from hallucination_guard.schemas import GroundedOutput


@pytest.fixture
def domain() -> GeneralDomain:
    return GeneralDomain()


def test_confidence_threshold_is_in_range(domain: GeneralDomain) -> None:
    assert 0.0 < domain.confidence_threshold <= 1.0


def test_is_valid_source_accepts_https(domain: GeneralDomain) -> None:
    assert domain.is_valid_source("https://example.com/article") is True


def test_is_valid_source_rejects_plain_http(domain: GeneralDomain) -> None:
    assert domain.is_valid_source("http://example.com") is False


def test_is_valid_source_rejects_data_url(domain: GeneralDomain) -> None:
    assert domain.is_valid_source("data:text/html,foo") is False


def test_is_valid_source_rejects_empty_string(domain: GeneralDomain) -> None:
    assert domain.is_valid_source("") is False


def test_is_valid_source_rejects_https_without_host(domain: GeneralDomain) -> None:
    assert domain.is_valid_source("https://") is False


def test_critic_prompt_is_non_empty(domain: GeneralDomain) -> None:
    prompt = domain.critic_prompt()
    assert isinstance(prompt, str)
    assert prompt.strip() != ""


def test_output_schema_returns_grounded_output_class(domain: GeneralDomain) -> None:
    schema = domain.output_schema()
    assert isinstance(schema, type)
    assert issubclass(schema, BaseModel)
    assert schema is GroundedOutput


def test_system_prompt_is_non_empty(domain: GeneralDomain) -> None:
    prompt = domain.system_prompt()
    assert isinstance(prompt, str)
    assert prompt.strip() != ""


def test_system_prompt_and_critic_prompt_are_distinct(domain: GeneralDomain) -> None:
    assert domain.system_prompt() != domain.critic_prompt()


def test_format_retry_directive_appends_fix_instruction(
    domain: GeneralDomain,
) -> None:
    directive = RetryDirective(fix_instruction="Add a citation URL.")
    prompt = domain.format_retry_directive("BASE", directive)
    assert prompt.startswith("BASE")
    assert "--- retry directive ---" in prompt
    assert "Add a citation URL." in prompt


def test_format_retry_directive_emits_forbidden_claims_bullets(
    domain: GeneralDomain,
) -> None:
    directive = RetryDirective(
        fix_instruction="Avoid contradictions.",
        forbidden_claims=["claim ALPHA", "claim BETA"],
    )
    prompt = domain.format_retry_directive("BASE", directive)
    assert "Do not repeat" in prompt
    assert "- claim ALPHA" in prompt
    assert "- claim BETA" in prompt


def test_format_retry_directive_omits_forbidden_section_when_empty(
    domain: GeneralDomain,
) -> None:
    directive = RetryDirective(fix_instruction="Raise confidence.")
    prompt = domain.format_retry_directive("BASE", directive)
    assert "Do not repeat" not in prompt


def test_default_locale_is_english() -> None:
    domain = GeneralDomain()
    assert domain.locale == "en"
    assert "research assistant" in domain.system_prompt().lower()


def test_japanese_locale_swaps_system_prompt() -> None:
    domain = GeneralDomain(locale="ja")
    prompt = domain.system_prompt()
    # "リサーチアシスタント" is the JA-specific phrasing not present in EN.
    assert "リサーチアシスタント" in prompt


def test_japanese_locale_swaps_critic_prompt() -> None:
    domain = GeneralDomain(locale="ja")
    prompt = domain.critic_prompt()
    assert "ファクトチェッカー" in prompt
    # Verdict markers stay in ASCII so structured-output parsing is unaffected.
    assert "verdict=PASS" in prompt
    assert "verdict=FAIL" in prompt


def test_japanese_locale_retry_directive_uses_japanese_header() -> None:
    domain = GeneralDomain(locale="ja")
    directive = RetryDirective(fix_instruction="出典URLを追加してください。")
    prompt = domain.format_retry_directive("BASE", directive)
    assert prompt.startswith("BASE")
    assert "--- 再試行指示 ---" in prompt
    assert "出典URLを追加してください。" in prompt
    # No leakage from the English variant.
    assert "--- retry directive ---" not in prompt


def test_japanese_locale_retry_directive_lists_forbidden_claims() -> None:
    domain = GeneralDomain(locale="ja")
    directive = RetryDirective(
        fix_instruction="矛盾を避けてください。",
        forbidden_claims=["主張アルファ", "主張ベータ"],
    )
    prompt = domain.format_retry_directive("BASE", directive)
    assert "却下済み主張" in prompt
    assert "- 主張アルファ" in prompt
    assert "- 主張ベータ" in prompt


def test_retry_locale_matches_constructor_locale() -> None:
    """``retry_locale()`` mirrors the locale used at construction so the
    hint builder picks the matching ``INSTRUCTION_MAPS`` entry."""
    assert GeneralDomain().retry_locale() == "en"
    assert GeneralDomain(locale="en").retry_locale() == "en"
    assert GeneralDomain(locale="ja").retry_locale() == "ja"
