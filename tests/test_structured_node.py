"""Tests for :mod:`hallucination_guard.nodes.structured_node`."""

from __future__ import annotations

from typing import Optional

import pytest
from pydantic import BaseModel

from hallucination_guard.domain.general import GeneralDomain
from hallucination_guard.exceptions import GraphError
from hallucination_guard.nodes.structured_node import StructuredNode
from hallucination_guard.schemas import Claim, GroundedOutput
from hallucination_guard.state import FailReason, GraphState


class FakeStructuredLLM:
    """In-memory ``StructuredLLM`` that records call args and returns ``payload``."""

    def __init__(self, payload: BaseModel) -> None:
        self.payload = payload
        self.captured_system: str = ""
        self.captured_user: str = ""
        self.captured_schema: Optional[type[BaseModel]] = None

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        self.captured_system = system
        self.captured_user = user
        self.captured_schema = schema
        return self.payload


def test_populates_research_output_from_llm_response() -> None:
    payload = GroundedOutput(
        claims=[Claim(text="x", confidence=0.9, sources=["https://e.com"])]
    )
    llm = FakeStructuredLLM(payload=payload)
    node = StructuredNode(GeneralDomain(), llm)

    result = node(GraphState(user_query="hi"))

    assert result.research_output is payload


def test_passes_domain_schema_to_llm() -> None:
    llm = FakeStructuredLLM(payload=GroundedOutput())
    node = StructuredNode(GeneralDomain(), llm)
    node(GraphState(user_query="q"))
    assert llm.captured_schema is GroundedOutput


def test_passes_user_query_through() -> None:
    llm = FakeStructuredLLM(payload=GroundedOutput())
    node = StructuredNode(GeneralDomain(), llm)
    node(GraphState(user_query="what is X?"))
    assert llm.captured_user == "what is X?"


def test_initial_attempt_does_not_inject_retry_directive() -> None:
    llm = FakeStructuredLLM(payload=GroundedOutput())
    node = StructuredNode(GeneralDomain(), llm)
    node(GraphState(user_query="q", retry_count=0))
    assert "retry directive" not in llm.captured_system


def test_retry_attempt_injects_fix_instruction_for_no_source() -> None:
    llm = FakeStructuredLLM(payload=GroundedOutput())
    node = StructuredNode(GeneralDomain(), llm)
    node(
        GraphState(
            user_query="q",
            retry_count=1,
            fail_reason=FailReason.NO_SOURCE,
        )
    )
    assert "retry directive" in llm.captured_system
    # English-locale fix_instruction for NO_SOURCE references "source URL".
    assert "source URL" in llm.captured_system


def test_retry_attempt_uses_japanese_fix_instruction_when_domain_locale_is_ja() -> None:
    """``StructuredNode`` forwards ``domain.retry_locale()`` to the hint builder."""
    llm = FakeStructuredLLM(payload=GroundedOutput())
    node = StructuredNode(GeneralDomain(locale="ja"), llm)
    node(
        GraphState(
            user_query="q",
            retry_count=1,
            fail_reason=FailReason.NO_SOURCE,
        )
    )
    assert "再試行指示" in llm.captured_system
    assert "出典URL" in llm.captured_system


def test_retry_with_critic_rejected_lists_forbidden_claims_in_prompt() -> None:
    llm = FakeStructuredLLM(payload=GroundedOutput())
    node = StructuredNode(GeneralDomain(), llm)
    node(
        GraphState(
            user_query="q",
            retry_count=1,
            fail_reason=FailReason.CRITIC_REJECTED,
            fail_history=[
                "critic_rejected:claim ALPHA",
                "low_confidence:irrelevant",
            ],
        )
    )
    assert "claim ALPHA" in llm.captured_system
    # The raw "critic_rejected:" prefix must NOT leak — only the stripped claim.
    assert "critic_rejected:claim ALPHA" not in llm.captured_system


def test_retry_with_fail_reason_but_no_forbidden_claims_skips_list() -> None:
    llm = FakeStructuredLLM(payload=GroundedOutput())
    node = StructuredNode(GeneralDomain(), llm)
    node(
        GraphState(
            user_query="q",
            retry_count=1,
            fail_reason=FailReason.LOW_CONFIDENCE,
            fail_history=["low_confidence:foo"],
        )
    )
    assert "Do not repeat" not in llm.captured_system


def test_does_not_mutate_input_state() -> None:
    llm = FakeStructuredLLM(payload=GroundedOutput())
    node = StructuredNode(GeneralDomain(), llm)
    state = GraphState(user_query="q")
    node(state)
    assert state.research_output is None


class FakeAsyncStructuredLLM:
    """In-memory ``AsyncStructuredLLM`` for exercising ``acall``."""

    def __init__(self, payload: BaseModel) -> None:
        self.payload = payload
        self.captured_system: str = ""
        self.captured_user: str = ""
        self.captured_schema: Optional[type[BaseModel]] = None

    async def agenerate(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        self.captured_system = system
        self.captured_user = user
        self.captured_schema = schema
        return self.payload


async def test_acall_invokes_async_client() -> None:
    payload = GroundedOutput(
        claims=[Claim(text="x", confidence=0.9, sources=["https://e.com"])]
    )
    llm = FakeAsyncStructuredLLM(payload=payload)
    node = StructuredNode(GeneralDomain(), llm)

    result = await node.acall(GraphState(user_query="q"))

    assert result.research_output is payload
    assert llm.captured_user == "q"
    assert llm.captured_schema is GroundedOutput


async def test_acall_with_sync_client_still_works() -> None:
    """``acall`` should work with sync clients too so a single async-mode
    graph can host CPU-only and IO-bound nodes side by side."""
    payload = GroundedOutput()
    llm = FakeStructuredLLM(payload=payload)
    node = StructuredNode(GeneralDomain(), llm)

    result = await node.acall(GraphState(user_query="q"))

    assert result.research_output is payload


def test_sync_call_with_async_client_raises() -> None:
    """Calling ``__call__`` on a node bound to an async client must fail
    loudly — otherwise LangGraph would hand the caller a coroutine in
    ``research_output``, which the FactCheckGate could not consume."""
    llm = FakeAsyncStructuredLLM(payload=GroundedOutput())
    node = StructuredNode(GeneralDomain(), llm)
    with pytest.raises(GraphError, match="async client"):
        node(GraphState(user_query="q"))


async def test_acall_injects_retry_directive_on_retry() -> None:
    llm = FakeAsyncStructuredLLM(payload=GroundedOutput())
    node = StructuredNode(GeneralDomain(), llm)
    await node.acall(
        GraphState(
            user_query="q",
            retry_count=1,
            fail_reason=FailReason.NO_SOURCE,
        )
    )
    assert "retry directive" in llm.captured_system
