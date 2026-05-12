"""CriticNode — independent judge that detects contradictions / unsupported claims.

Runs after :class:`FactCheckGate` passes. On a PASS verdict the node marks
the run as successful and stores a serialized snapshot of the output in
``final_output``. On a FAIL verdict it records each rejected claim in
``fail_history`` under the ``critic_rejected:`` prefix so the next
:class:`RetryHintBuilder.build` call can surface them as forbidden claims.

The node accepts either a sync :class:`JudgeLLM` or an async
:class:`AsyncJudgeLLM`. When an async client is injected, calling the node
synchronously raises; the graph must drive it via :meth:`acall`.
"""

from __future__ import annotations

from hallucination_guard.domain.base import DomainConfig
from hallucination_guard.exceptions import GraphError
from hallucination_guard.llm.protocols import AsyncJudgeLLM, JudgeLLM
from hallucination_guard.schemas import CriticVerdict
from hallucination_guard.state import FailReason, GraphState


class CriticNode:
    """Run an independent critic over the structured output."""

    def __init__(
        self, domain: DomainConfig, llm: JudgeLLM | AsyncJudgeLLM
    ) -> None:
        self.domain = domain
        self.llm = llm
        self._is_async = isinstance(llm, AsyncJudgeLLM) and not isinstance(
            llm, JudgeLLM
        )

    def __call__(self, state: GraphState) -> GraphState:
        if self._is_async:
            raise GraphError(
                "CriticNode is bound to an async client; drive it via "
                "Graph.arun() / Graph.astream() or call acall() directly."
            )
        content = self._extract_content(state)
        assert isinstance(self.llm, JudgeLLM)
        verdict = self.llm.judge(
            system=self.domain.critic_prompt(),
            content=content,
        )
        return self._apply_verdict(state, verdict, content)

    async def acall(self, state: GraphState) -> GraphState:
        """Async counterpart of :meth:`__call__`.

        Awaits ``ajudge`` when the injected client is async; otherwise falls
        back to the sync ``judge`` so a single node implementation can drive
        both modes through :meth:`Graph.astream`.
        """
        content = self._extract_content(state)
        if self._is_async:
            assert isinstance(self.llm, AsyncJudgeLLM)
            verdict = await self.llm.ajudge(
                system=self.domain.critic_prompt(),
                content=content,
            )
        else:
            assert isinstance(self.llm, JudgeLLM)
            verdict = self.llm.judge(
                system=self.domain.critic_prompt(),
                content=content,
            )
        return self._apply_verdict(state, verdict, content)

    @staticmethod
    def _extract_content(state: GraphState) -> str:
        output = state.research_output
        if output is None:
            raise GraphError("CriticNode received empty research_output")
        if hasattr(output, "model_dump_json"):
            return str(output.model_dump_json())
        return str(output)

    @staticmethod
    def _apply_verdict(
        state: GraphState, verdict: CriticVerdict, content: str
    ) -> GraphState:
        if verdict.verdict == "PASS":
            return state.with_update(
                critic_result="PASS",
                is_success=True,
                final_output=content,
                error_message=None,
            )
        additions = [
            f"{FailReason.CRITIC_REJECTED.value}:{claim}"
            for claim in verdict.rejected_claims
        ]
        return state.with_update(
            critic_result="FAIL",
            fail_reason=FailReason.CRITIC_REJECTED,
            fail_history=state.fail_history + additions,
        )
