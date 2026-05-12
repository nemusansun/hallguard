"""End-to-end tests for :mod:`hallucination_guard.graph`.

Exercises the full pipeline via scripted fakes so retry routing, exhaustion
handling, and the success path can be observed without any real LLM call.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Iterable

import pytest
from pydantic import BaseModel

from hallucination_guard.domain.general import GeneralDomain
from hallucination_guard.graph import Graph
from hallucination_guard.schemas import Claim, CriticVerdict, GroundedOutput
from hallucination_guard.state import FailReason, GraphState


class ScriptedStructuredLLM:
    """Returns successive outputs from a fixed script; loops the final value."""

    def __init__(self, outputs: Iterable[GroundedOutput]) -> None:
        self._outputs = list(outputs)
        if not self._outputs:
            raise ValueError("ScriptedStructuredLLM needs at least one output")
        self.calls: list[dict[str, str]] = []

    def generate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        self.calls.append({"system": system, "user": user})
        idx = min(len(self.calls) - 1, len(self._outputs) - 1)
        return self._outputs[idx]


class ScriptedJudgeLLM:
    """Returns successive verdicts from a fixed script; loops the final value."""

    def __init__(self, verdicts: Iterable[CriticVerdict]) -> None:
        self._verdicts = list(verdicts)
        if not self._verdicts:
            raise ValueError("ScriptedJudgeLLM needs at least one verdict")
        self.calls: list[dict[str, str]] = []

    def judge(self, *, system: str, content: str) -> CriticVerdict:
        self.calls.append({"system": system, "content": content})
        idx = min(len(self.calls) - 1, len(self._verdicts) - 1)
        return self._verdicts[idx]


def _good_claim() -> Claim:
    return Claim(
        text="Water boils at 100C at sea level",
        confidence=0.95,
        sources=["https://example.com/physics"],
    )


def _low_confidence_claim() -> Claim:
    return Claim(text="Unsure", confidence=0.2, sources=["https://example.com/a"])


def _bad_source_claim() -> Claim:
    return Claim(text="Tea cures cancer", confidence=0.9, sources=["http://shady"])


def test_success_path_on_first_attempt() -> None:
    structured = ScriptedStructuredLLM([GroundedOutput(claims=[_good_claim()])])
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    result = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    ).run("query")

    assert result.is_success is True
    assert result.retry_count == 0
    assert result.gate_result == "PASS"
    assert result.critic_result == "PASS"
    assert result.final_output is not None
    assert result.error_message is None
    assert len(structured.calls) == 1
    assert len(judge.calls) == 1


def test_low_confidence_retry_then_success() -> None:
    structured = ScriptedStructuredLLM(
        [
            GroundedOutput(claims=[_low_confidence_claim()]),
            GroundedOutput(claims=[_good_claim()]),
        ]
    )
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    result = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    ).run("query")

    assert result.is_success is True
    assert result.retry_count == 1
    assert result.critic_result == "PASS"
    assert len(structured.calls) == 2
    assert len(judge.calls) == 1  # only fired after the gate passed


def test_critic_rejection_then_retry_then_success() -> None:
    structured = ScriptedStructuredLLM(
        [
            GroundedOutput(claims=[_good_claim()]),
            GroundedOutput(claims=[_good_claim()]),
        ]
    )
    judge = ScriptedJudgeLLM(
        [
            CriticVerdict(verdict="FAIL", rejected_claims=["bogus"]),
            CriticVerdict(verdict="PASS"),
        ]
    )

    result = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    ).run("query")

    assert result.is_success is True
    assert result.retry_count == 1
    assert "critic_rejected:bogus" in result.fail_history
    assert len(judge.calls) == 2


def test_max_retries_reached_routes_to_error_output() -> None:
    structured = ScriptedStructuredLLM(
        [GroundedOutput(claims=[_low_confidence_claim()])]
    )
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    result = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
        max_retries=2,
    ).run("query")

    assert result.is_success is False
    assert result.retry_count == 2
    assert result.final_output is None
    assert result.error_message is not None
    assert "max_retries (2) reached" in result.error_message
    assert "low_confidence" in result.error_message
    # Initial attempt + max_retries retries = 3 generations
    assert len(structured.calls) == 3
    # Judge never runs because the gate never passes
    assert judge.calls == []


def test_no_source_failure_routes_to_retry() -> None:
    structured = ScriptedStructuredLLM(
        [
            GroundedOutput(claims=[_bad_source_claim()]),
            GroundedOutput(claims=[_good_claim()]),
        ]
    )
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    result = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    ).run("query")

    assert result.is_success is True
    assert result.retry_count == 1
    assert any(
        entry.startswith(FailReason.NO_SOURCE.value + ":")
        for entry in result.fail_history
    )


def test_retry_directive_reaches_structured_node_via_system_prompt() -> None:
    structured = ScriptedStructuredLLM(
        [
            GroundedOutput(claims=[_low_confidence_claim()]),
            GroundedOutput(claims=[_good_claim()]),
        ]
    )
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    ).run("query")

    first_system = structured.calls[0]["system"]
    second_system = structured.calls[1]["system"]
    assert "retry directive" not in first_system
    assert "retry directive" in second_system
    # Raw fail_history entries must never leak into the prompt
    assert "low_confidence:Unsure" not in second_system


def test_critic_rejected_claims_become_forbidden_on_retry() -> None:
    structured = ScriptedStructuredLLM(
        [
            GroundedOutput(claims=[_good_claim()]),
            GroundedOutput(claims=[_good_claim()]),
        ]
    )
    judge = ScriptedJudgeLLM(
        [
            CriticVerdict(verdict="FAIL", rejected_claims=["forbidden-A"]),
            CriticVerdict(verdict="PASS"),
        ]
    )

    Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    ).run("query")

    retry_system = structured.calls[1]["system"]
    assert "forbidden-A" in retry_system
    assert "Do not repeat" in retry_system


def test_negative_max_retries_rejected() -> None:
    with pytest.raises(ValueError):
        Graph(
            domain=GeneralDomain(),
            structured_llm=ScriptedStructuredLLM(
                [GroundedOutput(claims=[_good_claim()])]
            ),
            judge_llm=ScriptedJudgeLLM([CriticVerdict(verdict="PASS")]),
            max_retries=-1,
        )


def test_max_retries_zero_initial_failure_routes_to_error_immediately() -> None:
    structured = ScriptedStructuredLLM(
        [GroundedOutput(claims=[_low_confidence_claim()])]
    )
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    result = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
        max_retries=0,
    ).run("query")

    assert result.is_success is False
    assert result.retry_count == 0
    assert result.error_message is not None
    assert "max_retries (0) reached" in result.error_message
    # Initial attempt only — no retry
    assert len(structured.calls) == 1
    assert judge.calls == []


def test_max_retries_zero_initial_success_still_works() -> None:
    structured = ScriptedStructuredLLM([GroundedOutput(claims=[_good_claim()])])
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    result = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
        max_retries=0,
    ).run("query")

    assert result.is_success is True
    assert result.retry_count == 0
    assert result.error_message is None


def test_max_retries_zero_critic_rejection_routes_to_error() -> None:
    structured = ScriptedStructuredLLM([GroundedOutput(claims=[_good_claim()])])
    judge = ScriptedJudgeLLM(
        [CriticVerdict(verdict="FAIL", rejected_claims=["bogus"])]
    )

    result = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
        max_retries=0,
    ).run("query")

    assert result.is_success is False
    assert result.retry_count == 0
    assert result.error_message is not None
    assert "critic_rejected" in result.error_message
    assert len(structured.calls) == 1
    assert len(judge.calls) == 1


def test_run_returns_graphstate_instance() -> None:
    structured = ScriptedStructuredLLM([GroundedOutput(claims=[_good_claim()])])
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    result = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    ).run("query")

    assert isinstance(result, GraphState)


def test_checkpointer_persists_state_across_run_and_get_state() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    structured = ScriptedStructuredLLM(
        [
            GroundedOutput(claims=[_low_confidence_claim()]),
            GroundedOutput(claims=[_good_claim()]),
        ]
    )
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
        checkpointer=InMemorySaver(),
    )
    result = graph.run("query", thread_id="t1")

    assert result.is_success is True
    assert result.retry_count == 1
    # research_output keeps the duck-typed .claims attribute through checkpointing
    assert result.research_output is not None
    assert hasattr(result.research_output, "claims")

    persisted = graph.get_state("t1")
    assert persisted.user_query == "query"
    assert persisted.is_success is True
    assert persisted.retry_count == 1
    assert persisted.final_output == result.final_output


def test_checkpointer_requires_thread_id() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    structured = ScriptedStructuredLLM([GroundedOutput(claims=[_good_claim()])])
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
        checkpointer=InMemorySaver(),
    )
    with pytest.raises(ValueError, match="thread_id"):
        graph.run("query")


def test_get_state_without_checkpointer_raises() -> None:
    structured = ScriptedStructuredLLM([GroundedOutput(claims=[_good_claim()])])
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    )
    with pytest.raises(RuntimeError, match="checkpointer"):
        graph.get_state("nope")


def test_stream_yields_node_events_in_execution_order() -> None:
    structured = ScriptedStructuredLLM(
        [
            GroundedOutput(claims=[_low_confidence_claim()]),
            GroundedOutput(claims=[_good_claim()]),
        ]
    )
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    )
    events = list(graph.stream("q"))

    nodes = [ev.node for ev in events]
    assert nodes == ["structured", "factcheck", "retry", "structured", "factcheck", "critic"]
    # Cumulative state must be coherent: gate flips FAIL then PASS, critic PASS at end
    assert events[1].state.gate_result == "FAIL"
    assert events[2].state.retry_count == 1
    assert events[-1].state.is_success is True
    assert events[-1].state.critic_result == "PASS"


def test_stream_final_state_matches_run() -> None:
    def _structured() -> ScriptedStructuredLLM:
        return ScriptedStructuredLLM(
            [
                GroundedOutput(claims=[_low_confidence_claim()]),
                GroundedOutput(claims=[_good_claim()]),
            ]
        )

    def _judge() -> ScriptedJudgeLLM:
        return ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    streamed_final = list(
        Graph(
            domain=GeneralDomain(),
            structured_llm=_structured(),
            judge_llm=_judge(),
        ).stream("q")
    )[-1].state

    invoked = Graph(
        domain=GeneralDomain(),
        structured_llm=_structured(),
        judge_llm=_judge(),
    ).run("q")

    assert streamed_final.is_success == invoked.is_success
    assert streamed_final.retry_count == invoked.retry_count
    assert streamed_final.final_output == invoked.final_output


def test_stream_preserves_research_output_pydantic_instance() -> None:
    structured = ScriptedStructuredLLM([GroundedOutput(claims=[_good_claim()])])
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    events = list(
        Graph(
            domain=GeneralDomain(),
            structured_llm=structured,
            judge_llm=judge,
        ).stream("q")
    )

    # research_output must keep its concrete .claims attribute through streaming
    # — the same risk model_dump() created in Graph.run().
    last = events[-1].state
    assert last.research_output is not None
    assert hasattr(last.research_output, "claims")


def test_stream_requires_thread_id_when_checkpointer_configured() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=ScriptedStructuredLLM(
            [GroundedOutput(claims=[_good_claim()])]
        ),
        judge_llm=ScriptedJudgeLLM([CriticVerdict(verdict="PASS")]),
        checkpointer=InMemorySaver(),
    )
    with pytest.raises(ValueError, match="thread_id"):
        list(graph.stream("q"))


async def test_astream_yields_node_events_in_execution_order() -> None:
    structured = ScriptedStructuredLLM(
        [
            GroundedOutput(claims=[_low_confidence_claim()]),
            GroundedOutput(claims=[_good_claim()]),
        ]
    )
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    )
    events = [ev async for ev in graph.astream("q")]

    nodes = [ev.node for ev in events]
    assert nodes == [
        "structured",
        "factcheck",
        "retry",
        "structured",
        "factcheck",
        "critic",
    ]
    assert events[-1].state.is_success is True
    assert events[-1].state.critic_result == "PASS"


async def test_astream_final_state_matches_run() -> None:
    def _structured() -> ScriptedStructuredLLM:
        return ScriptedStructuredLLM(
            [
                GroundedOutput(claims=[_low_confidence_claim()]),
                GroundedOutput(claims=[_good_claim()]),
            ]
        )

    def _judge() -> ScriptedJudgeLLM:
        return ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    streamed_final = [
        ev
        async for ev in Graph(
            domain=GeneralDomain(),
            structured_llm=_structured(),
            judge_llm=_judge(),
        ).astream("q")
    ][-1].state

    invoked = Graph(
        domain=GeneralDomain(),
        structured_llm=_structured(),
        judge_llm=_judge(),
    ).run("q")

    assert streamed_final.is_success == invoked.is_success
    assert streamed_final.retry_count == invoked.retry_count
    assert streamed_final.final_output == invoked.final_output


async def test_astream_preserves_research_output_pydantic_instance() -> None:
    structured = ScriptedStructuredLLM([GroundedOutput(claims=[_good_claim()])])
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    events = [
        ev
        async for ev in Graph(
            domain=GeneralDomain(),
            structured_llm=structured,
            judge_llm=judge,
        ).astream("q")
    ]

    last = events[-1].state
    assert last.research_output is not None
    assert hasattr(last.research_output, "claims")


async def test_astream_requires_thread_id_when_checkpointer_configured() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=ScriptedStructuredLLM(
            [GroundedOutput(claims=[_good_claim()])]
        ),
        judge_llm=ScriptedJudgeLLM([CriticVerdict(verdict="PASS")]),
        checkpointer=InMemorySaver(),
    )
    with pytest.raises(ValueError, match="thread_id"):
        async for _ in graph.astream("q"):
            pass


def test_checkpointer_isolates_threads() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    # First thread succeeds on the first try; second thread fails out via retries.
    structured = ScriptedStructuredLLM(
        [GroundedOutput(claims=[_good_claim()])]
    )
    judge = ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
        checkpointer=saver,
    )
    a = graph.run("query A", thread_id="thread-a")
    b = graph.run("query B", thread_id="thread-b")

    assert a.user_query == "query A"
    assert b.user_query == "query B"
    assert graph.get_state("thread-a").user_query == "query A"
    assert graph.get_state("thread-b").user_query == "query B"


class AsyncScriptedStructuredLLM:
    """Async-native scripted ``AsyncStructuredLLM``; mirrors ``ScriptedStructuredLLM``."""

    def __init__(self, outputs: Iterable[GroundedOutput]) -> None:
        self._outputs = list(outputs)
        if not self._outputs:
            raise ValueError("AsyncScriptedStructuredLLM needs at least one output")
        self.calls: list[dict[str, str]] = []

    async def agenerate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        self.calls.append({"system": system, "user": user})
        idx = min(len(self.calls) - 1, len(self._outputs) - 1)
        return self._outputs[idx]


class AsyncScriptedJudgeLLM:
    """Async-native scripted ``AsyncJudgeLLM``; mirrors ``ScriptedJudgeLLM``."""

    def __init__(self, verdicts: Iterable[CriticVerdict]) -> None:
        self._verdicts = list(verdicts)
        if not self._verdicts:
            raise ValueError("AsyncScriptedJudgeLLM needs at least one verdict")
        self.calls: list[dict[str, str]] = []

    async def ajudge(self, *, system: str, content: str) -> CriticVerdict:
        self.calls.append({"system": system, "content": content})
        idx = min(len(self.calls) - 1, len(self._verdicts) - 1)
        return self._verdicts[idx]


def test_is_async_flag_reflects_client_types() -> None:
    sync_graph = Graph(
        domain=GeneralDomain(),
        structured_llm=ScriptedStructuredLLM(
            [GroundedOutput(claims=[_good_claim()])]
        ),
        judge_llm=ScriptedJudgeLLM([CriticVerdict(verdict="PASS")]),
    )
    assert sync_graph.is_async is False

    async_graph = Graph(
        domain=GeneralDomain(),
        structured_llm=AsyncScriptedStructuredLLM(
            [GroundedOutput(claims=[_good_claim()])]
        ),
        judge_llm=AsyncScriptedJudgeLLM([CriticVerdict(verdict="PASS")]),
    )
    assert async_graph.is_async is True


def test_async_mode_rejects_sync_run() -> None:
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=AsyncScriptedStructuredLLM(
            [GroundedOutput(claims=[_good_claim()])]
        ),
        judge_llm=AsyncScriptedJudgeLLM([CriticVerdict(verdict="PASS")]),
    )
    with pytest.raises(RuntimeError, match="async LLM client"):
        graph.run("q")


def test_async_mode_rejects_sync_stream() -> None:
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=AsyncScriptedStructuredLLM(
            [GroundedOutput(claims=[_good_claim()])]
        ),
        judge_llm=AsyncScriptedJudgeLLM([CriticVerdict(verdict="PASS")]),
    )
    with pytest.raises(RuntimeError, match="async LLM client"):
        list(graph.stream("q"))


def test_mixed_async_structured_with_sync_judge_is_async_mode() -> None:
    """Async-mode kicks in if *any* client is async-only — the graph must
    not silently expose a sync surface that would await nothing."""
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=AsyncScriptedStructuredLLM(
            [GroundedOutput(claims=[_good_claim()])]
        ),
        judge_llm=ScriptedJudgeLLM([CriticVerdict(verdict="PASS")]),
    )
    assert graph.is_async is True
    with pytest.raises(RuntimeError):
        graph.run("q")


async def test_arun_with_async_clients_completes_success_path() -> None:
    structured = AsyncScriptedStructuredLLM(
        [GroundedOutput(claims=[_good_claim()])]
    )
    judge = AsyncScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    )
    result = await graph.arun("q")

    assert result.is_success is True
    assert result.retry_count == 0
    assert result.critic_result == "PASS"
    assert len(structured.calls) == 1
    assert len(judge.calls) == 1


async def test_arun_with_async_clients_drives_retry_loop() -> None:
    structured = AsyncScriptedStructuredLLM(
        [
            GroundedOutput(claims=[_low_confidence_claim()]),
            GroundedOutput(claims=[_good_claim()]),
        ]
    )
    judge = AsyncScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    )
    result = await graph.arun("q")

    assert result.is_success is True
    assert result.retry_count == 1
    assert len(structured.calls) == 2


async def test_arun_with_sync_clients_matches_run() -> None:
    """``arun`` must work with sync clients too so callers already living
    inside an event loop don't have to switch APIs."""
    def _structured() -> ScriptedStructuredLLM:
        return ScriptedStructuredLLM(
            [
                GroundedOutput(claims=[_low_confidence_claim()]),
                GroundedOutput(claims=[_good_claim()]),
            ]
        )

    def _judge() -> ScriptedJudgeLLM:
        return ScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    sync_result = Graph(
        domain=GeneralDomain(),
        structured_llm=_structured(),
        judge_llm=_judge(),
    ).run("q")

    async_result = await Graph(
        domain=GeneralDomain(),
        structured_llm=_structured(),
        judge_llm=_judge(),
    ).arun("q")

    assert sync_result.is_success == async_result.is_success
    assert sync_result.retry_count == async_result.retry_count
    assert sync_result.final_output == async_result.final_output


async def test_astream_with_async_clients_yields_full_event_sequence() -> None:
    structured = AsyncScriptedStructuredLLM(
        [
            GroundedOutput(claims=[_low_confidence_claim()]),
            GroundedOutput(claims=[_good_claim()]),
        ]
    )
    judge = AsyncScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    )
    events = [ev async for ev in graph.astream("q")]
    nodes = [ev.node for ev in events]

    assert nodes == [
        "structured",
        "factcheck",
        "retry",
        "structured",
        "factcheck",
        "critic",
    ]
    assert events[-1].state.is_success is True


async def test_astream_with_async_clients_preserves_pydantic_research_output() -> None:
    structured = AsyncScriptedStructuredLLM(
        [GroundedOutput(claims=[_good_claim()])]
    )
    judge = AsyncScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    events = [
        ev
        async for ev in Graph(
            domain=GeneralDomain(),
            structured_llm=structured,
            judge_llm=judge,
        ).astream("q")
    ]
    last = events[-1].state
    assert last.research_output is not None
    assert hasattr(last.research_output, "claims")


async def test_arun_with_async_clients_exhausts_to_error_output() -> None:
    structured = AsyncScriptedStructuredLLM(
        [GroundedOutput(claims=[_low_confidence_claim()])]
    )
    judge = AsyncScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    result = await Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
        max_retries=1,
    ).arun("q")

    assert result.is_success is False
    assert result.retry_count == 1
    assert result.error_message is not None
    assert "max_retries (1) reached" in result.error_message
    # Initial attempt + 1 retry = 2 generations
    assert len(structured.calls) == 2
    assert judge.calls == []


class BlockingAsyncStructuredLLM:
    """Async structured client that sleeps long enough to be cancellable.

    The ``entered`` / ``cancelled`` flags let cancellation tests verify
    that ``CancelledError`` actually unwinds through ``agenerate`` rather
    than the caller raising before the LLM call started.
    """

    def __init__(self, *, sleep_seconds: float = 10.0) -> None:
        self._sleep = sleep_seconds
        self.entered = False
        self.cancelled = False

    async def agenerate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        self.entered = True
        try:
            await asyncio.sleep(self._sleep)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return GroundedOutput(claims=[_good_claim()])


class BlockingAsyncJudgeLLM:
    async def ajudge(self, *, system: str, content: str) -> CriticVerdict:
        await asyncio.sleep(10.0)
        return CriticVerdict(verdict="PASS")


async def test_arun_is_cancellable_via_wait_for_timeout() -> None:
    """A blocking async LLM call must unwind cleanly when the outer
    ``asyncio.wait_for`` fires its timeout, surfacing ``TimeoutError``
    to the caller rather than leaking a coroutine or swallowing it."""
    structured = BlockingAsyncStructuredLLM(sleep_seconds=5.0)
    judge = BlockingAsyncJudgeLLM()

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    )
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await asyncio.wait_for(graph.arun("q"), timeout=0.05)

    assert structured.entered is True
    assert structured.cancelled is True


async def test_arun_propagates_explicit_cancellation() -> None:
    """Cancelling the task wrapping ``arun`` must propagate
    ``CancelledError`` and let the structured client observe the
    cancellation inside its ``await``."""
    structured = BlockingAsyncStructuredLLM(sleep_seconds=5.0)
    judge = BlockingAsyncJudgeLLM()

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    )
    task = asyncio.create_task(graph.arun("q"))
    # Give the task a moment to actually enter agenerate before cancelling.
    for _ in range(20):
        await asyncio.sleep(0)
        if structured.entered:
            break
    assert structured.entered is True

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert structured.cancelled is True


async def test_astream_cancelled_consumer_unwinds_blocking_call() -> None:
    """A task that pulls from ``astream`` and is then cancelled must
    propagate the cancellation all the way through the LLM coroutine,
    not leave it pending or swallow it inside the generator."""
    structured = BlockingAsyncStructuredLLM(sleep_seconds=5.0)
    judge = BlockingAsyncJudgeLLM()

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    )

    async def _consume() -> None:
        async for _event in graph.astream("q"):
            return  # never reached — the first node blocks

    consumer = asyncio.create_task(_consume())
    for _ in range(20):
        await asyncio.sleep(0)
        if structured.entered:
            break
    assert structured.entered is True

    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert structured.cancelled is True


async def test_completed_arun_is_idempotent_against_cancel_after_done() -> None:
    """If the task has already completed, ``task.cancel()`` returns False
    and the result is still observable — i.e. completed work isn't lost
    just because a stale cancellation arrives."""
    structured = AsyncScriptedStructuredLLM(
        [GroundedOutput(claims=[_good_claim()])]
    )
    judge = AsyncScriptedJudgeLLM([CriticVerdict(verdict="PASS")])

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    )
    task = asyncio.create_task(graph.arun("q"))
    result = await task

    assert task.cancel() is False  # already completed; cancel is a no-op
    assert result.is_success is True


class _StressAsyncStructuredLLM:
    """Query-keyed async structured fake for high-concurrency fan-out tests.

    Each query string carries its own behavior in its prefix, and a
    per-query attempt counter keeps the response sequence deterministic
    even when many ``arun`` calls share this instance:

    - ``"easy:..."``     -> a confident, well-cited claim on every attempt
    - ``"recovers:..."`` -> low confidence on attempt 1, confident on >=2
    - ``"fails:..."``    -> low confidence on every attempt

    Because each ``arun`` call uses a distinct query string, the
    ``Counter`` entries are partitioned by call: there is no shared key
    that two concurrent invocations both increment, so the simulator's
    determinism does not depend on scheduler ordering.

    ``current_in_flight`` / ``peak_in_flight`` expose how many
    ``agenerate`` invocations were simultaneously inside the ``await``,
    which the test suite uses to confirm that fan-out actually
    overlapped under the semaphore bound.
    """

    def __init__(self) -> None:
        self._attempts: Counter[str] = Counter()
        self.current_in_flight = 0
        self.peak_in_flight = 0

    async def agenerate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        self.current_in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.current_in_flight)
        try:
            await asyncio.sleep(0)
            self._attempts[user] += 1
            attempt = self._attempts[user]

            if user.startswith("easy:"):
                claim = _good_claim()
            elif user.startswith("recovers:"):
                claim = _good_claim() if attempt >= 2 else _low_confidence_claim()
            elif user.startswith("fails:"):
                claim = _low_confidence_claim()
            else:
                raise AssertionError(f"unrecognized stress query: {user!r}")
            return GroundedOutput(claims=[claim])
        finally:
            self.current_in_flight -= 1


class _StressAsyncJudgeLLM:
    """Always-PASS async judge for stress tests; yields once for interleaving."""

    async def ajudge(self, *, system: str, content: str) -> CriticVerdict:
        await asyncio.sleep(0)
        return CriticVerdict(verdict="PASS")


def _expected_outcome(query: str) -> tuple[bool, int]:
    """Return ``(is_success, retry_count)`` for a stress query, matching the
    fake LLM's per-prefix behavior under ``max_retries=3``."""
    if query.startswith("easy:"):
        return True, 0
    if query.startswith("recovers:"):
        return True, 1
    if query.startswith("fails:"):
        return False, 3
    raise AssertionError(query)


async def test_arun_high_concurrency_matches_serial_results() -> None:
    """Fanning many ``arun`` calls through a single ``Graph`` under a
    bounded semaphore must yield, per query, the exact same final state
    that a serial sweep produces. The contract under test is state
    isolation between concurrent invocations: every ``ainvoke`` builds
    its own ``GraphState`` from the supplied query, so retry counters,
    fail histories, and final outputs must not bleed across calls.
    """
    queries = (
        [f"easy:{i}" for i in range(24)]
        + [f"recovers:{i}" for i in range(24)]
        + [f"fails:{i}" for i in range(16)]
    )

    # Serial baseline — one fresh graph + fake per case so the per-query
    # counter is never reused across queries.
    serial_results: dict[str, tuple[bool, int]] = {}
    for q in queries:
        baseline = await Graph(
            domain=GeneralDomain(),
            structured_llm=_StressAsyncStructuredLLM(),
            judge_llm=_StressAsyncJudgeLLM(),
        ).arun(q)
        assert baseline.user_query == q
        serial_results[q] = (baseline.is_success, baseline.retry_count)
        assert serial_results[q] == _expected_outcome(q)

    # Concurrent run — single Graph, single fake, semaphore-bounded.
    structured = _StressAsyncStructuredLLM()
    judge = _StressAsyncJudgeLLM()
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    )

    sem = asyncio.Semaphore(32)

    async def _run(q: str) -> GraphState:
        async with sem:
            return await graph.arun(q)

    results = await asyncio.gather(*(_run(q) for q in queries))

    for q, r in zip(queries, results):
        assert r.user_query == q
        assert (r.is_success, r.retry_count) == serial_results[q]

    # Each query appears exactly once in the result set — no cross-talk.
    assert [r.user_query for r in results] == queries

    # Actual overlap occurred and the semaphore was never breached.
    assert structured.peak_in_flight >= 2
    assert structured.peak_in_flight <= 32


async def test_arun_concurrent_calls_do_not_share_fail_history() -> None:
    """Interleave 32 always-failing queries with 32 always-succeeding
    queries through one shared ``Graph``. If any internal state were
    aliased across calls, a failing query's accumulated ``fail_history``
    could surface on a succeeding query's result. Each side must come
    back with the exact history its own pipeline produced.
    """
    queries = [f"easy:{i}" for i in range(32)] + [f"fails:{i}" for i in range(32)]

    structured = _StressAsyncStructuredLLM()
    judge = _StressAsyncJudgeLLM()
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=structured,
        judge_llm=judge,
    )

    sem = asyncio.Semaphore(32)

    async def _run(q: str) -> GraphState:
        async with sem:
            return await graph.arun(q)

    results = await asyncio.gather(*(_run(q) for q in queries))

    for q, r in zip(queries, results):
        assert r.user_query == q
        if q.startswith("easy:"):
            assert r.is_success is True
            assert r.retry_count == 0
            assert r.fail_history == []
        else:
            assert r.is_success is False
            assert r.retry_count == 3
            # Initial attempt + 3 retries each contribute one LOW_CONFIDENCE entry.
            assert len(r.fail_history) == 4
            for entry in r.fail_history:
                assert entry.startswith(FailReason.LOW_CONFIDENCE.value + ":")

    assert structured.peak_in_flight >= 2
    assert structured.peak_in_flight <= 32
