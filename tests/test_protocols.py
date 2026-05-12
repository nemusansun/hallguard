"""Tests for :mod:`hallucination_guard.llm.protocols`.

Exercises the runtime-checkable Protocol classes so the graph layer can
trust ``isinstance(client, AsyncStructuredLLM)`` to discriminate async-only
clients from sync ones at construction time.
"""

from __future__ import annotations

from pydantic import BaseModel

from hallucination_guard.llm.protocols import (
    AsyncJudgeLLM,
    AsyncStructuredLLM,
    JudgeLLM,
    StructuredLLM,
)
from hallucination_guard.schemas import CriticVerdict, GroundedOutput


class SyncStructured:
    def generate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        return GroundedOutput()


class AsyncStructured:
    async def agenerate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        return GroundedOutput()


class DualStructured:
    def generate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        return GroundedOutput()

    async def agenerate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        return GroundedOutput()


class SyncJudge:
    def judge(self, *, system: str, content: str) -> CriticVerdict:
        return CriticVerdict(verdict="PASS")


class AsyncJudge:
    async def ajudge(self, *, system: str, content: str) -> CriticVerdict:
        return CriticVerdict(verdict="PASS")


def test_sync_structured_satisfies_only_sync_protocol() -> None:
    client = SyncStructured()
    assert isinstance(client, StructuredLLM)
    assert not isinstance(client, AsyncStructuredLLM)


def test_async_structured_satisfies_only_async_protocol() -> None:
    client = AsyncStructured()
    assert isinstance(client, AsyncStructuredLLM)
    assert not isinstance(client, StructuredLLM)


def test_dual_structured_satisfies_both_protocols() -> None:
    """A class exposing both methods passes both checks.

    The graph constructor must then prefer the sync surface (see
    ``_is_async_client``) so existing callers don't silently switch modes.
    """
    client = DualStructured()
    assert isinstance(client, StructuredLLM)
    assert isinstance(client, AsyncStructuredLLM)


def test_sync_judge_satisfies_only_sync_protocol() -> None:
    client = SyncJudge()
    assert isinstance(client, JudgeLLM)
    assert not isinstance(client, AsyncJudgeLLM)


def test_async_judge_satisfies_only_async_protocol() -> None:
    client = AsyncJudge()
    assert isinstance(client, AsyncJudgeLLM)
    assert not isinstance(client, JudgeLLM)


def test_unrelated_object_satisfies_neither_protocol() -> None:
    assert not isinstance(object(), StructuredLLM)
    assert not isinstance(object(), AsyncStructuredLLM)
    assert not isinstance(object(), JudgeLLM)
    assert not isinstance(object(), AsyncJudgeLLM)
