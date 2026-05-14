"""Tests for :mod:`hallucination_guard.domain.medical`."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from hallucination_guard.domain.medical import _ALLOWED_HOSTS, MedicalDomain
from hallucination_guard.retry.directive import RetryDirective
from hallucination_guard.schemas import GroundedOutput
from hallucination_guard.state import FailReason


@pytest.fixture
def domain() -> MedicalDomain:
    return MedicalDomain()


def test_confidence_threshold_is_strict(domain: MedicalDomain) -> None:
    assert domain.confidence_threshold >= 0.9


def test_allows_pubmed_https(domain: MedicalDomain) -> None:
    assert domain.is_valid_source("https://pubmed.ncbi.nlm.nih.gov/12345") is True


def test_allows_who_https(domain: MedicalDomain) -> None:
    assert domain.is_valid_source("https://www.who.int/news/foo") is True


def test_rejects_unknown_host_even_over_https(domain: MedicalDomain) -> None:
    assert domain.is_valid_source("https://example.com/article") is False


def test_rejects_plain_http_pubmed(domain: MedicalDomain) -> None:
    assert domain.is_valid_source("http://pubmed.ncbi.nlm.nih.gov/12345") is False


def test_rejects_empty_string(domain: MedicalDomain) -> None:
    assert domain.is_valid_source("") is False


def test_host_match_is_case_insensitive(domain: MedicalDomain) -> None:
    assert domain.is_valid_source("https://WWW.WHO.INT/page") is True


def test_critic_prompt_mentions_clinical_safety(domain: MedicalDomain) -> None:
    prompt = domain.critic_prompt()
    assert "clinical" in prompt.lower() or "medical" in prompt.lower()


def test_output_schema_is_grounded_output(domain: MedicalDomain) -> None:
    schema = domain.output_schema()
    assert isinstance(schema, type)
    assert issubclass(schema, BaseModel)
    assert schema is GroundedOutput


def test_system_prompt_mentions_citations(domain: MedicalDomain) -> None:
    prompt = domain.system_prompt().lower()
    assert "cit" in prompt  # "citation" / "cite"
    assert "source" in prompt or "evidence" in prompt


def test_format_retry_directive_uses_medical_marker(
    domain: MedicalDomain,
) -> None:
    directive = RetryDirective(fix_instruction="Add a citation URL.")
    prompt = domain.format_retry_directive("BASE", directive)
    assert prompt.startswith("BASE")
    # The medical marker distinguishes the strict domain from GeneralDomain
    # so demos can show the prompts differ between domains.
    assert "(medical)" in prompt
    assert "Add a citation URL." in prompt


def test_format_retry_directive_reinforces_citation_requirement(
    domain: MedicalDomain,
) -> None:
    directive = RetryDirective(fix_instruction="Raise confidence above 0.95.")
    prompt = domain.format_retry_directive("BASE", directive)
    lowered = prompt.lower()
    assert "peer-reviewed" in lowered or "institutional" in lowered


def test_format_retry_directive_lists_forbidden_clinical_claims(
    domain: MedicalDomain,
) -> None:
    directive = RetryDirective(
        fix_instruction="Avoid contradictions.",
        forbidden_claims=["aspirin cures diabetes"],
    )
    prompt = domain.format_retry_directive("BASE", directive)
    assert "previously-rejected" in prompt
    assert "- aspirin cures diabetes" in prompt


# Friendly source names enumerated in MedicalDomain.format_retry_directive
# (the "(PubMed, WHO, CDC, Cochrane, NEJM)" list). Each maps to substrings
# that must appear in at least one host of _ALLOWED_HOSTS. Keeping these in
# lock-step catches the case where _ALLOWED_HOSTS gains/loses a source but
# the retry template — which is what the LLM actually sees — drifts.
_FRIENDLY_TO_HOST_TOKENS: dict[str, tuple[str, ...]] = {
    # PubMed lives under NCBI, so the umbrella token covers both
    # pubmed.ncbi.nlm.nih.gov and bare ncbi.nlm.nih.gov entries.
    "pubmed": ("pubmed", "ncbi"),
    "who": ("who.int",),
    "cdc": ("cdc.gov",),
    "cochrane": ("cochrane",),
    "nejm": ("nejm",),
}


def test_retry_template_friendly_names_each_have_matching_host(
    domain: MedicalDomain,
) -> None:
    """Every friendly name named in the template must have a real host."""
    template = domain.format_retry_directive(
        "BASE", RetryDirective(fix_instruction="x")
    ).lower()
    for friendly, host_tokens in _FRIENDLY_TO_HOST_TOKENS.items():
        assert friendly in template, (
            f"retry template no longer mentions {friendly!r}; update "
            f"_FRIENDLY_TO_HOST_TOKENS or restore the mention"
        )
        assert any(
            any(token in host for host in _ALLOWED_HOSTS)
            for token in host_tokens
        ), (
            f"template mentions {friendly!r} but no host in _ALLOWED_HOSTS "
            f"matches any of {host_tokens!r}"
        )


def test_every_allowed_host_is_named_in_retry_template(
    domain: MedicalDomain,
) -> None:
    """Every host in _ALLOWED_HOSTS must be covered by a named source."""
    template = domain.format_retry_directive(
        "BASE", RetryDirective(fix_instruction="x")
    ).lower()
    covered_tokens = {
        token
        for tokens in _FRIENDLY_TO_HOST_TOKENS.values()
        for token in tokens
    }
    for host in _ALLOWED_HOSTS:
        assert any(token in host for token in covered_tokens), (
            f"host {host!r} is in _ALLOWED_HOSTS but not represented by "
            f"any token in _FRIENDLY_TO_HOST_TOKENS — extend the mapping "
            f"and the template if you added a new source"
        )
    for friendly in _FRIENDLY_TO_HOST_TOKENS:
        assert friendly in template, (
            f"mapped friendly name {friendly!r} is not mentioned in the "
            f"retry template anymore"
        )


def test_default_locale_is_english() -> None:
    domain = MedicalDomain()
    assert domain.locale == "en"
    # English prompt mentions the role title in ASCII; the JA variant uses
    # 「臨床研究アシスタント」 instead.
    assert "clinical research assistant" in domain.system_prompt().lower()


def test_japanese_locale_swaps_system_prompt() -> None:
    domain = MedicalDomain(locale="ja")
    prompt = domain.system_prompt()
    assert "臨床研究アシスタント" in prompt


def test_japanese_locale_swaps_critic_prompt() -> None:
    domain = MedicalDomain(locale="ja")
    prompt = domain.critic_prompt()
    assert "臨床" in prompt
    # Verdict markers must remain ASCII so structured-output parsing works.
    assert "verdict=PASS" in prompt
    assert "verdict=FAIL" in prompt


def test_japanese_locale_retry_directive_keeps_medical_marker() -> None:
    domain = MedicalDomain(locale="ja")
    directive = RetryDirective(fix_instruction="出典URLを追加してください。")
    prompt = domain.format_retry_directive("BASE", directive)
    assert prompt.startswith("BASE")
    # The "(medical)" suffix is the cross-locale marker that distinguishes
    # MedicalDomain from GeneralDomain at a glance.
    assert "(medical)" in prompt
    assert "--- 再試行指示 (medical) ---" in prompt
    assert "--- retry directive (medical) ---" not in prompt


def test_japanese_locale_retry_directive_keeps_source_brand_names() -> None:
    """Friendly source names (PubMed/WHO/CDC/Cochrane/NEJM) stay in
    ASCII even in the Japanese variant — they are proper nouns that match
    the host-allow-list and must not be transliterated.
    """
    domain = MedicalDomain(locale="ja")
    prompt = domain.format_retry_directive(
        "BASE", RetryDirective(fix_instruction="x")
    )
    for brand in ("PubMed", "WHO", "CDC", "Cochrane", "NEJM"):
        assert brand in prompt


def test_japanese_locale_retry_directive_lists_forbidden_clinical_claims() -> None:
    domain = MedicalDomain(locale="ja")
    directive = RetryDirective(
        fix_instruction="矛盾を避けてください。",
        forbidden_claims=["アスピリンは糖尿病を治す"],
    )
    prompt = domain.format_retry_directive("BASE", directive)
    assert "却下済み臨床主張" in prompt
    assert "- アスピリンは糖尿病を治す" in prompt


@pytest.mark.parametrize("reason", list(FailReason))
def test_retry_instruction_covers_every_fail_reason(reason: FailReason) -> None:
    """Each :class:`FailReason` resolves to a non-empty instruction in both locales."""
    en = MedicalDomain(locale="en").retry_instruction(reason)
    ja = MedicalDomain(locale="ja").retry_instruction(reason)
    assert en and ja


@pytest.mark.parametrize("reason", list(FailReason))
def test_retry_instruction_differs_between_locales(reason: FailReason) -> None:
    """Each ``FailReason`` has a distinct phrase per locale.

    Guards against the two locale maps collapsing into a single language.
    """
    en = MedicalDomain(locale="en").retry_instruction(reason)
    ja = MedicalDomain(locale="ja").retry_instruction(reason)
    assert en != ja


def test_no_source_retry_instruction_names_a_medical_source_en() -> None:
    """The English ``NO_SOURCE`` phrase steers the LLM toward the
    allow-listed brands — this is what differentiates MedicalDomain from
    GeneralDomain at the retry-instruction layer."""
    phrase = MedicalDomain(locale="en").retry_instruction(FailReason.NO_SOURCE)
    assert "PubMed" in phrase or "peer-reviewed" in phrase
