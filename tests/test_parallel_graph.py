"""Integration tests for parallel (fan-out / fan-in) graph execution."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from hallucination_guard.domain.general import GeneralDomain
from hallucination_guard.graph import Graph
from hallucination_guard.nodes.aggregator import AggregatorNode, default_merge
from hallucination_guard.schemas import Claim, CriticVerdict, GroundedOutput
from hallucination_guard.state import GraphState


# --- Fake LLMs -----------------------------------------------------------


class _FakeStructuredLLM:
    """Returns a GroundedOutput whose claim text encodes the LLM identity."""

    def __init__(self, identity: str, confidence: float = 0.9) -> None:
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


class _FakeJudgeLLM:
    """Always passes."""

    def judge(self, *, system: str, content: str) -> CriticVerdict:
        return CriticVerdict(verdict="PASS")


class _FailThenPassJudgeLLM:
    """Fails on the first call, then passes."""

    def __init__(self) -> None:
        self._call_count = 0

    def judge(self, *, system: str, content: str) -> CriticVerdict:
        self._call_count += 1
        if self._call_count == 1:
            return CriticVerdict(
                verdict="FAIL",
                rejected_claims=["bad claim"],
                reason="first attempt rejected",
            )
        return CriticVerdict(verdict="PASS")


class _LowConfidenceLLM:
    """Returns low-confidence output to trigger FactCheckGate failure."""

    def __init__(self, identity: str) -> None:
        self._identity = identity

    def generate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> GroundedOutput:
        return GroundedOutput(
            claims=[
                Claim(
                    text=f"{self._identity}: {user}",
                    confidence=0.1,
                    sources=["https://en.wikipedia.org/wiki/Example"],
                )
            ]
        )


# --- AggregatorNode unit tests -------------------------------------------


def test_default_merge_combines_claims() -> None:
    outputs = [
        GroundedOutput(claims=[Claim(text="A", confidence=0.9)]),
        GroundedOutput(claims=[Claim(text="B", confidence=0.8)]),
    ]
    merged = default_merge(outputs)
    assert len(merged.claims) == 2
    assert {c.text for c in merged.claims} == {"A", "B"}


def test_aggregator_takes_last_n() -> None:
    """AggregatorNode takes last N items from branch_outputs."""
    out1 = GroundedOutput(claims=[Claim(text="old", confidence=0.9)])
    out2 = GroundedOutput(claims=[Claim(text="new_a", confidence=0.9)])
    out3 = GroundedOutput(claims=[Claim(text="new_b", confidence=0.9)])

    state = GraphState(
        user_query="q",
        branch_outputs=[out1, out2, out3],  # out1 is from a previous round
    )
    agg = AggregatorNode(num_researchers=2)
    result = agg(state)
    assert len(result.research_output.claims) == 2
    texts = {c.text for c in result.research_output.claims}
    assert texts == {"new_a", "new_b"}


def test_aggregator_custom_merge_fn() -> None:
    """Custom merge_fn replaces default_merge."""

    def pick_first(outputs: list[Any]) -> Any:
        return outputs[0]

    out_a = GroundedOutput(claims=[Claim(text="A", confidence=0.9)])
    out_b = GroundedOutput(claims=[Claim(text="B", confidence=0.9)])
    state = GraphState(user_query="q", branch_outputs=[out_a, out_b])

    agg = AggregatorNode(num_researchers=2, merge_fn=pick_first)
    result = agg(state)
    assert result.research_output.claims[0].text == "A"


# --- Parallel Graph integration tests ------------------------------------


def test_parallel_graph_properties() -> None:
    """Parallel graph exposes is_parallel=True and num_researchers."""
    domain = GeneralDomain()
    llms = [_FakeStructuredLLM("A"), _FakeStructuredLLM("B")]
    graph = Graph(domain=domain, structured_llm=llms, judge_llm=_FakeJudgeLLM())
    assert graph.is_parallel is True
    assert graph.num_researchers == 2


def test_single_llm_is_not_parallel() -> None:
    domain = GeneralDomain()
    graph = Graph(
        domain=domain,
        structured_llm=_FakeStructuredLLM("A"),
        judge_llm=_FakeJudgeLLM(),
    )
    assert graph.is_parallel is False
    assert graph.num_researchers == 1


def test_parallel_run_merges_all_claims() -> None:
    """Both researchers' claims appear in the final output."""
    domain = GeneralDomain()
    llm_a = _FakeStructuredLLM("researcher_A")
    llm_b = _FakeStructuredLLM("researcher_B")
    graph = Graph(
        domain=domain, structured_llm=[llm_a, llm_b], judge_llm=_FakeJudgeLLM()
    )
    result = graph.run("test query")
    assert result.is_success is True
    assert result.final_output is not None
    # branch_outputs should have entries from both researchers
    assert len(result.branch_outputs) == 2


def test_parallel_fail_history_not_lost() -> None:
    """fail_history from gate failure must survive across parallel branches.

    This is the core scenario the reducer pattern was introduced to fix.
    """
    domain = GeneralDomain()
    # Both produce low confidence -> FactCheckGate FAIL
    llm_a = _LowConfidenceLLM("A")
    llm_b = _LowConfidenceLLM("B")
    graph = Graph(
        domain=domain,
        structured_llm=[llm_a, llm_b],
        judge_llm=_FakeJudgeLLM(),
        max_retries=1,
    )
    result = graph.run("test")
    assert result.is_success is False
    # fail_history should have entries from BOTH branches' aggregated
    # output, across retries. Must not be empty.
    assert len(result.fail_history) > 0


def test_parallel_retry_re_dispatches() -> None:
    """After a critic FAIL, retry re-fans-out to all researchers."""
    domain = GeneralDomain()
    llm_a = _FakeStructuredLLM("A")
    llm_b = _FakeStructuredLLM("B")
    judge = _FailThenPassJudgeLLM()
    graph = Graph(
        domain=domain,
        structured_llm=[llm_a, llm_b],
        judge_llm=judge,
        max_retries=3,
    )
    result = graph.run("test retry")
    assert result.is_success is True
    # Should have 4 branch_outputs: 2 from round 0 + 2 from round 1
    assert len(result.branch_outputs) == 4
    # fail_history should have the critic_rejected entry from round 0
    assert any("critic_rejected" in h for h in result.fail_history)


def test_parallel_max_retries_zero() -> None:
    """max_retries=0 with parallel: first FAIL goes straight to ErrorOutput."""
    domain = GeneralDomain()
    llm_a = _LowConfidenceLLM("A")
    llm_b = _LowConfidenceLLM("B")
    graph = Graph(
        domain=domain,
        structured_llm=[llm_a, llm_b],
        judge_llm=_FakeJudgeLLM(),
        max_retries=0,
    )
    result = graph.run("test")
    assert result.is_success is False
    assert result.error_message is not None
    assert "max_retries (0) reached" in result.error_message


def test_parallel_stream_yields_events() -> None:
    """stream() in parallel mode yields dispatch/structured/aggregate nodes."""
    domain = GeneralDomain()
    llm_a = _FakeStructuredLLM("A")
    llm_b = _FakeStructuredLLM("B")
    graph = Graph(
        domain=domain, structured_llm=[llm_a, llm_b], judge_llm=_FakeJudgeLLM()
    )
    events = list(graph.stream("test"))
    node_names = [e.node for e in events]
    # dispatch is a no-op (no state changes) so it may not emit a stream
    # event.  The key nodes that must appear are: structured (x2),
    # aggregate, factcheck, critic.
    assert "aggregate" in node_names
    assert node_names.count("structured") == 2
    # Final state must be success
    assert events[-1].state.is_success is True


def test_parallel_empty_list_raises() -> None:
    """Passing an empty list of structured_llm must raise."""
    import pytest

    domain = GeneralDomain()
    with pytest.raises(ValueError, match="must not be empty"):
        Graph(domain=domain, structured_llm=[], judge_llm=_FakeJudgeLLM())


def test_parallel_custom_merge_strategy() -> None:
    """merge_strategy kwarg overrides default claim concatenation."""
    domain = GeneralDomain()
    llm_a = _FakeStructuredLLM("A")
    llm_b = _FakeStructuredLLM("B")

    def first_only(outputs: list[Any]) -> Any:
        return outputs[0]

    graph = Graph(
        domain=domain,
        structured_llm=[llm_a, llm_b],
        judge_llm=_FakeJudgeLLM(),
        merge_strategy=first_only,
    )
    result = graph.run("test")
    assert result.is_success is True
