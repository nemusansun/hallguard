"""Tests for :mod:`hallucination_guard.nodes.critic_node`."""

from __future__ import annotations

import pytest

from hallucination_guard.domain.general import GeneralDomain
from hallucination_guard.exceptions import GraphError
from hallucination_guard.nodes.critic_node import CriticNode
from hallucination_guard.schemas import Claim, CriticVerdict, GroundedOutput
from hallucination_guard.state import FailReason, GraphState


class FakeJudgeLLM:
    """In-memory ``JudgeLLM`` that returns a preset verdict and records args."""

    def __init__(self, verdict: CriticVerdict) -> None:
        self.verdict = verdict
        self.captured_system: str = ""
        self.captured_content: str = ""

    def judge(self, *, system: str, content: str) -> CriticVerdict:
        self.captured_system = system
        self.captured_content = content
        return self.verdict


def _state_with_output() -> GraphState:
    return GraphState(
        user_query="q",
        research_output=GroundedOutput(
            claims=[Claim(text="ok", confidence=0.9, sources=["https://e.com"])]
        ),
    )


def test_pass_verdict_marks_success_and_sets_final_output() -> None:
    llm = FakeJudgeLLM(verdict=CriticVerdict(verdict="PASS"))
    node = CriticNode(GeneralDomain(), llm)

    result = node(_state_with_output())

    assert result.critic_result == "PASS"
    assert result.is_success is True
    assert result.final_output is not None
    assert "ok" in result.final_output
    assert result.error_message is None


def test_fail_verdict_appends_rejected_claims_with_prefix() -> None:
    llm = FakeJudgeLLM(
        verdict=CriticVerdict(
            verdict="FAIL",
            rejected_claims=["claim A", "claim B"],
        )
    )
    node = CriticNode(GeneralDomain(), llm)

    result = node(_state_with_output())

    assert result.critic_result == "FAIL"
    assert result.fail_reason == FailReason.CRITIC_REJECTED
    assert result.fail_history == [
        "critic_rejected:claim A",
        "critic_rejected:claim B",
    ]
    assert result.is_success is False


def test_fail_verdict_with_no_rejected_claims_still_marks_failure() -> None:
    llm = FakeJudgeLLM(verdict=CriticVerdict(verdict="FAIL"))
    node = CriticNode(GeneralDomain(), llm)

    result = node(_state_with_output())

    assert result.critic_result == "FAIL"
    assert result.fail_reason == FailReason.CRITIC_REJECTED
    assert result.fail_history == []


def test_uses_domain_critic_prompt() -> None:
    llm = FakeJudgeLLM(verdict=CriticVerdict(verdict="PASS"))
    domain = GeneralDomain()
    node = CriticNode(domain, llm)
    node(_state_with_output())
    assert llm.captured_system == domain.critic_prompt()


def test_appends_to_existing_fail_history() -> None:
    llm = FakeJudgeLLM(
        verdict=CriticVerdict(verdict="FAIL", rejected_claims=["new claim"])
    )
    node = CriticNode(GeneralDomain(), llm)
    state = GraphState(
        user_query="q",
        research_output=GroundedOutput(
            claims=[Claim(text="ok", confidence=0.9, sources=["https://e.com"])]
        ),
        fail_history=["low_confidence:earlier"],
    )

    result = node(state)

    assert result.fail_history == [
        "low_confidence:earlier",
        "critic_rejected:new claim",
    ]


def test_raises_when_research_output_is_missing() -> None:
    llm = FakeJudgeLLM(verdict=CriticVerdict(verdict="PASS"))
    node = CriticNode(GeneralDomain(), llm)
    state = GraphState(user_query="q", research_output=None)

    with pytest.raises(GraphError):
        node(state)


def test_does_not_mutate_input_state() -> None:
    llm = FakeJudgeLLM(
        verdict=CriticVerdict(verdict="FAIL", rejected_claims=["x"])
    )
    node = CriticNode(GeneralDomain(), llm)
    state = _state_with_output()
    node(state)
    assert state.fail_history == []
    assert state.fail_reason is None
    assert state.is_success is False


class FakeAsyncJudgeLLM:
    """In-memory ``AsyncJudgeLLM`` for exercising ``acall``."""

    def __init__(self, verdict: CriticVerdict) -> None:
        self.verdict = verdict
        self.captured_system: str = ""
        self.captured_content: str = ""

    async def ajudge(self, *, system: str, content: str) -> CriticVerdict:
        self.captured_system = system
        self.captured_content = content
        return self.verdict


async def test_acall_with_async_pass_marks_success() -> None:
    llm = FakeAsyncJudgeLLM(verdict=CriticVerdict(verdict="PASS"))
    node = CriticNode(GeneralDomain(), llm)

    result = await node.acall(_state_with_output())

    assert result.is_success is True
    assert result.critic_result == "PASS"
    assert result.final_output is not None


async def test_acall_with_async_fail_records_rejected_claims() -> None:
    llm = FakeAsyncJudgeLLM(
        verdict=CriticVerdict(
            verdict="FAIL", rejected_claims=["bad-A", "bad-B"]
        )
    )
    node = CriticNode(GeneralDomain(), llm)

    result = await node.acall(_state_with_output())

    assert result.critic_result == "FAIL"
    assert result.fail_reason == FailReason.CRITIC_REJECTED
    assert result.fail_history == [
        "critic_rejected:bad-A",
        "critic_rejected:bad-B",
    ]


async def test_acall_with_sync_client_still_works() -> None:
    llm = FakeJudgeLLM(verdict=CriticVerdict(verdict="PASS"))
    node = CriticNode(GeneralDomain(), llm)

    result = await node.acall(_state_with_output())

    assert result.is_success is True


def test_sync_call_with_async_client_raises() -> None:
    llm = FakeAsyncJudgeLLM(verdict=CriticVerdict(verdict="PASS"))
    node = CriticNode(GeneralDomain(), llm)
    with pytest.raises(GraphError, match="async client"):
        node(_state_with_output())
