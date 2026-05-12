"""StructuredNode — schema-constrained LLM call.

The first attempt uses the domain's base system prompt only. On retries
(``retry_count > 0`` with a recorded ``fail_reason``) the node builds a
:class:`RetryDirective` via :class:`RetryHintBuilder` and hands it to the
domain's ``format_retry_directive`` to assemble the final prompt — this is
the **only** path by which retry information reaches the LLM. Raw
``fail_history`` strings are never forwarded.

The node accepts either a sync :class:`StructuredLLM` or an async
:class:`AsyncStructuredLLM`. When an async client is injected, calling the
node synchronously raises; the graph must drive it via :meth:`acall`.
"""

from __future__ import annotations

from hallucination_guard.domain.base import DomainConfig
from hallucination_guard.exceptions import GraphError
from hallucination_guard.llm.protocols import AsyncStructuredLLM, StructuredLLM
from hallucination_guard.retry.hint_builder import RetryHintBuilder
from hallucination_guard.state import GraphState


class StructuredNode:
    """Generate a domain-typed answer from the user's query."""

    def __init__(
        self,
        domain: DomainConfig,
        llm: StructuredLLM | AsyncStructuredLLM,
    ) -> None:
        self.domain = domain
        self.llm = llm
        self._is_async = isinstance(llm, AsyncStructuredLLM) and not isinstance(
            llm, StructuredLLM
        )

    def __call__(self, state: GraphState) -> GraphState:
        if self._is_async:
            raise GraphError(
                "StructuredNode is bound to an async client; drive it via "
                "Graph.arun() / Graph.astream() or call acall() directly."
            )
        system = self._build_system(state)
        assert isinstance(self.llm, StructuredLLM)
        output = self.llm.generate(
            system=system,
            user=state.user_query,
            schema=self.domain.output_schema(),
        )
        return state.with_update(research_output=output)

    async def acall(self, state: GraphState) -> GraphState:
        """Async counterpart of :meth:`__call__`.

        Awaits ``agenerate`` when the injected client is async; otherwise
        falls back to the sync ``generate`` so a single node implementation
        can drive both modes through :meth:`Graph.astream`.
        """
        system = self._build_system(state)
        if self._is_async:
            assert isinstance(self.llm, AsyncStructuredLLM)
            output = await self.llm.agenerate(
                system=system,
                user=state.user_query,
                schema=self.domain.output_schema(),
            )
        else:
            assert isinstance(self.llm, StructuredLLM)
            output = self.llm.generate(
                system=system,
                user=state.user_query,
                schema=self.domain.output_schema(),
            )
        return state.with_update(research_output=output)

    def _build_system(self, state: GraphState) -> str:
        base = self.domain.system_prompt()
        if state.retry_count > 0 and state.fail_reason is not None:
            directive = RetryHintBuilder.build(
                state, locale=self.domain.retry_locale()
            )
            return self.domain.format_retry_directive(base, directive)
        return base
