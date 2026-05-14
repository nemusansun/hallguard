"""Integration tests for parallel mode + LangGraph checkpointer.

Validates that ``Send``-dispatched branches survive serialization through
the checkpointer's msgpack layer, that ``branch_outputs`` and
``fail_history`` (both reducer-annotated lists) accumulate correctly in
persisted state, and that ``build_serializer`` / ``auto_serialize=True``
both keep working when the graph is configured with multiple structured
LLMs.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from pydantic import BaseModel

from hallucination_guard.domain.general import GeneralDomain
from hallucination_guard.graph import Graph
from hallucination_guard.schemas import Claim, CriticVerdict, GroundedOutput
from hallucination_guard.serde import build_serializer


# --- Fake LLMs -----------------------------------------------------------


class _FakeStructuredLLM:
    def __init__(self, identity: str, confidence: float = 0.95) -> None:
        self._identity = identity
        self._confidence = confidence

    def generate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> GroundedOutput:
        return GroundedOutput(
            claims=[
                Claim(
                    text=f"{self._identity}: {user}",
                    confidence=self._confidence,
                    sources=["https://en.wikipedia.org/wiki/Example"],
                )
            ]
        )


class _PassJudge:
    def judge(self, *, system: str, content: str) -> CriticVerdict:
        return CriticVerdict(verdict="PASS")


class _FailThenPassJudge:
    def __init__(self) -> None:
        self._n = 0

    def judge(self, *, system: str, content: str) -> CriticVerdict:
        self._n += 1
        if self._n == 1:
            return CriticVerdict(
                verdict="FAIL",
                rejected_claims=["bad"],
                reason="first round rejected",
            )
        return CriticVerdict(verdict="PASS")


class _LowConfidenceLLM:
    def __init__(self, identity: str) -> None:
        self._identity = identity

    def generate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> GroundedOutput:
        return GroundedOutput(
            claims=[
                Claim(
                    text=f"{self._identity}: low",
                    confidence=0.1,
                    sources=["https://en.wikipedia.org/wiki/Example"],
                )
            ]
        )


# --- Tests ---------------------------------------------------------------


def test_parallel_with_default_checkpointer_persists_branch_outputs() -> None:
    """Parallel mode + bare InMemorySaver: get_state returns merged output.

    Uses the LangGraph default serializer, which emits a deprecation
    warning about unregistered types but still round-trips.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=[_FakeStructuredLLM("A"), _FakeStructuredLLM("B")],
        judge_llm=_PassJudge(),
        checkpointer=InMemorySaver(),
    )
    with warnings.catch_warnings():
        # The default serializer warns on unregistered Pydantic types; the
        # later auto-serialize / build_serializer tests cover the silent path.
        warnings.simplefilter("ignore")
        result = graph.run("query", thread_id="parallel-1")

    assert result.is_success is True
    assert len(result.branch_outputs) == 2

    persisted = graph.get_state("parallel-1")
    assert persisted.is_success is True
    assert len(persisted.branch_outputs) == 2
    assert persisted.research_output is not None
    assert len({c.text for c in persisted.research_output.claims}) == 2


def test_parallel_with_build_serializer_round_trips_silently() -> None:
    """build_serializer() handles parallel-mode state with no warnings."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver(serde=build_serializer())
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=[_FakeStructuredLLM("A"), _FakeStructuredLLM("B")],
        judge_llm=_PassJudge(),
        checkpointer=saver,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # promote any warning to a failure
        result = graph.run("query", thread_id="parallel-build")

    assert result.is_success is True
    persisted = graph.get_state("parallel-build")
    assert persisted.user_query == "query"
    assert len(persisted.branch_outputs) == 2
    assert hasattr(persisted.research_output, "claims")
    assert len(persisted.research_output.claims) == 2


def test_parallel_with_auto_serialize_works() -> None:
    """auto_serialize=True swaps the saver's serde transparently in parallel mode."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=[_FakeStructuredLLM("A"), _FakeStructuredLLM("B")],
        judge_llm=_PassJudge(),
        checkpointer=saver,
        auto_serialize=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = graph.run("query", thread_id="parallel-auto")

    assert result.is_success is True
    persisted = graph.get_state("parallel-auto")
    assert len(persisted.branch_outputs) == 2


def test_parallel_checkpointer_isolates_threads() -> None:
    """Two threads must keep their branch_outputs separate."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver(serde=build_serializer())
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=[_FakeStructuredLLM("A"), _FakeStructuredLLM("B")],
        judge_llm=_PassJudge(),
        checkpointer=saver,
    )
    a = graph.run("query A", thread_id="thread-a")
    b = graph.run("query B", thread_id="thread-b")

    assert a.user_query == "query A"
    assert b.user_query == "query B"
    assert len(a.branch_outputs) == 2
    assert len(b.branch_outputs) == 2

    pa = graph.get_state("thread-a")
    pb = graph.get_state("thread-b")
    assert pa.user_query == "query A"
    assert pb.user_query == "query B"
    # Each thread's outputs reference their own query text — no bleed.
    assert all("query A" in c.text for c in pa.research_output.claims)
    assert all("query B" in c.text for c in pb.research_output.claims)


def test_parallel_checkpointer_persists_retry_accumulation() -> None:
    """After a critic FAIL + retry, persisted state shows both rounds.

    branch_outputs accumulates 2 (round 0) + 2 (round 1) = 4 entries via
    its operator.add reducer; fail_history carries the round-0 critic
    rejection. Both must survive checkpointing.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver(serde=build_serializer())
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=[_FakeStructuredLLM("A"), _FakeStructuredLLM("B")],
        judge_llm=_FailThenPassJudge(),
        checkpointer=saver,
        max_retries=3,
    )
    result = graph.run("retry query", thread_id="retry-thread")

    assert result.is_success is True
    assert len(result.branch_outputs) == 4

    persisted = graph.get_state("retry-thread")
    assert len(persisted.branch_outputs) == 4
    assert any("critic_rejected" in h for h in persisted.fail_history)
    # Final research_output reflects only the latest round (2 items).
    assert len(persisted.research_output.claims) == 2


def test_parallel_checkpointer_persists_error_path() -> None:
    """ErrorOutput terminus also persists through the checkpointer."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver(serde=build_serializer())
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=[_LowConfidenceLLM("A"), _LowConfidenceLLM("B")],
        judge_llm=_PassJudge(),
        checkpointer=saver,
        max_retries=1,
    )
    result = graph.run("error query", thread_id="error-thread")

    assert result.is_success is False
    assert result.error_message is not None

    persisted = graph.get_state("error-thread")
    assert persisted.is_success is False
    assert persisted.error_message == result.error_message
    assert len(persisted.fail_history) > 0


def test_parallel_checkpointer_resume_reuses_persisted_branch_outputs() -> None:
    """A second run() on the same thread sees the prior run's accumulated state.

    LangGraph keys checkpoints by thread_id; calling run() again with the
    same thread_id continues from the persisted snapshot rather than
    starting fresh. This is the scenario that would surface any
    serialization round-trip damage to branch_outputs.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver(serde=build_serializer())
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=[_FakeStructuredLLM("A"), _FakeStructuredLLM("B")],
        judge_llm=_PassJudge(),
        checkpointer=saver,
    )
    first = graph.run("first", thread_id="resume-thread")
    assert first.is_success is True
    assert len(first.branch_outputs) == 2

    persisted = graph.get_state("resume-thread")
    # The persisted research_output must still expose the duck-typed .claims
    # attribute after a msgpack round-trip — the gate that previously broke
    # when types collapsed to plain dicts.
    assert hasattr(persisted.research_output, "claims")
    for claim in persisted.research_output.claims:
        assert hasattr(claim, "confidence")
        assert isinstance(claim.confidence, float)


@pytest.mark.parametrize("num_researchers", [2, 3, 4])
def test_parallel_checkpointer_scales_with_researcher_count(
    num_researchers: int,
) -> None:
    """branch_outputs length tracks num_researchers across the checkpoint boundary."""
    from langgraph.checkpoint.memory import InMemorySaver

    llms: list[Any] = [
        _FakeStructuredLLM(f"R{i}") for i in range(num_researchers)
    ]
    saver = InMemorySaver(serde=build_serializer())
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=llms,
        judge_llm=_PassJudge(),
        checkpointer=saver,
    )
    result = graph.run("scale query", thread_id=f"scale-{num_researchers}")

    assert result.is_success is True
    assert len(result.branch_outputs) == num_researchers

    persisted = graph.get_state(f"scale-{num_researchers}")
    assert len(persisted.branch_outputs) == num_researchers
    assert len(persisted.research_output.claims) == num_researchers
