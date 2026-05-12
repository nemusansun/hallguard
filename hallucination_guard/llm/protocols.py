"""Protocols describing the LLM client surface used by nodes.

Concrete adapters (OpenAI, Mistral, vLLM, …) implement these protocols and
are injected into nodes at construction time. Tests inject in-memory fakes.
Defining the surface as :class:`typing.Protocol` keeps nodes free of any
direct vendor dependency.

Sync and async surfaces are kept as separate protocols with distinct method
names (``generate`` / ``agenerate``, ``judge`` / ``ajudge``) so that a single
adapter class can choose to implement one, the other, or both without method
collisions. Marking the protocols ``runtime_checkable`` lets the graph layer
decide at construction time whether the configured client is async — and
therefore whether the pipeline must be driven through the async entry points
— by a plain ``isinstance`` check.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from hallucination_guard.schemas import CriticVerdict


@runtime_checkable
class StructuredLLM(Protocol):
    """LLM client returning a Pydantic-validated structured output.

    Implementations MUST use ``temperature=0`` (or the closest deterministic
    setting available) and bind ``schema`` to the model call so that the
    return value can be parsed without post-processing.
    """

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        """Return an instance of ``schema`` produced by the LLM."""
        ...


@runtime_checkable
class JudgeLLM(Protocol):
    """LLM client acting as a critic and returning a :class:`CriticVerdict`."""

    def judge(
        self,
        *,
        system: str,
        content: str,
    ) -> CriticVerdict:
        """Return the critic's verdict for ``content``."""
        ...


@runtime_checkable
class AsyncStructuredLLM(Protocol):
    """Async counterpart of :class:`StructuredLLM`.

    Implementations expose ``agenerate`` rather than ``generate`` so a single
    class may implement both protocols without method collision. The graph
    layer dispatches on the presence of ``agenerate`` to decide whether the
    pipeline runs through async wrappers.
    """

    async def agenerate(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        """Return an instance of ``schema`` produced by the LLM."""
        ...


@runtime_checkable
class AsyncJudgeLLM(Protocol):
    """Async counterpart of :class:`JudgeLLM`.

    Exposes ``ajudge`` rather than ``judge`` for the same reason
    :class:`AsyncStructuredLLM` exposes ``agenerate``: distinct names let one
    adapter class implement both surfaces simultaneously.
    """

    async def ajudge(
        self,
        *,
        system: str,
        content: str,
    ) -> CriticVerdict:
        """Return the critic's verdict for ``content``."""
        ...
