"""End-to-end demo: a tiny research agent built on hallguard.

Runs the full StructuredNode → FactCheckGate → CriticNode pipeline against
in-memory fake LLMs so the example works without API keys. The fakes are
scripted to demonstrate both the success path (first attempt accepted) and
the retry path (a first-attempt low-confidence answer is rejected, the
retry directive nudges the LLM toward a confident citation, and the second
attempt passes both the gate and the critic).

Six demos are emitted:

1. ``Graph.run()`` — retry-then-success.
2. ``Graph.run()`` — exhausted retries falling back to ``ErrorOutput``.
3. ``Graph.stream()`` — per-node progress events from the synchronous API.
4. ``Graph.astream()`` — same progress events via the async API, driven by
   ``asyncio.run`` so the example stays a single ``python -m`` invocation.
5. ``Graph.arun()`` driven by a native :class:`AsyncStructuredLLM` client,
   showing how a vendor SDK exposing ``async`` methods plugs into the
   framework with no bridging.
6. Parallel research — two ``StructuredLLM`` clients fan out via ``Send``
   and the ``AggregatorNode`` merges their claims into a single output
   before the FactCheckGate runs.

Run me with:

    python -m examples.research_agent
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from hallucination_guard.domain.general import GeneralDomain
from hallucination_guard.graph import Graph
from hallucination_guard.schemas import Claim, CriticVerdict, GroundedOutput
from hallucination_guard.state import GraphState


class _DemoStructuredLLM:
    """Two-shot scripted LLM: low confidence first, confident on retry."""

    def __init__(self) -> None:
        self._step = 0

    def generate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        self._step += 1
        if self._step == 1:
            return GroundedOutput(
                claims=[
                    Claim(
                        text="The capital of France might be Paris",
                        confidence=0.55,
                        sources=["https://example.com/maybe"],
                    )
                ]
            )
        return GroundedOutput(
            claims=[
                Claim(
                    text="The capital of France is Paris",
                    confidence=0.98,
                    sources=["https://en.wikipedia.org/wiki/Paris"],
                )
            ]
        )


class _DemoJudgeLLM:
    """Always PASS — the gate is what rejects the first attempt here."""

    def judge(self, *, system: str, content: str) -> CriticVerdict:
        return CriticVerdict(verdict="PASS")


class _AsyncDemoStructuredLLM:
    """Async-native :class:`AsyncStructuredLLM` implementation.

    Mirrors vendor SDKs that expose only awaitable methods (httpx-backed
    clients, the ``AsyncOpenAI`` surface, etc.). The first attempt yields a
    low-confidence answer so the retry loop fires; the second clears the
    gate.
    """

    def __init__(self) -> None:
        self._step = 0

    async def agenerate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        # ``sleep(0)`` yields control to the event loop so the demo behaves
        # like a real awaitable network call without depending on time.
        await asyncio.sleep(0)
        self._step += 1
        if self._step == 1:
            return GroundedOutput(
                claims=[
                    Claim(
                        text="Paris is probably the capital of France",
                        confidence=0.5,
                        sources=["https://example.com/maybe"],
                    )
                ]
            )
        return GroundedOutput(
            claims=[
                Claim(
                    text="The capital of France is Paris",
                    confidence=0.97,
                    sources=["https://en.wikipedia.org/wiki/Paris"],
                )
            ]
        )


class _AsyncDemoJudgeLLM:
    """Async-native :class:`AsyncJudgeLLM` implementation: always PASS."""

    async def ajudge(self, *, system: str, content: str) -> CriticVerdict:
        await asyncio.sleep(0)
        return CriticVerdict(verdict="PASS")


class _PersonaStructuredLLM:
    """Returns a single confident claim tagged with a researcher persona.

    Used by the parallel demo so each fan-out branch contributes a
    distinguishable claim and the merged ``GroundedOutput`` visibly carries
    contributions from every researcher.
    """

    def __init__(self, persona: str, claim_text: str, source: str) -> None:
        self._persona = persona
        self._claim_text = claim_text
        self._source = source

    def generate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        return GroundedOutput(
            claims=[
                Claim(
                    text=f"[{self._persona}] {self._claim_text}",
                    confidence=0.95,
                    sources=[self._source],
                )
            ]
        )


def _report(result: GraphState) -> None:
    print(f"  success      : {result.is_success}")
    print(f"  retry_count  : {result.retry_count}")
    print(f"  gate_result  : {result.gate_result}")
    print(f"  critic_result: {result.critic_result}")
    if result.final_output:
        print(f"  final_output : {result.final_output}")
    if result.error_message:
        print(f"  error_message: {result.error_message}")
    if result.fail_history:
        print(f"  fail_history : {result.fail_history}")


def main() -> None:
    print("== demo 1: retry-then-success ==")
    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=_DemoStructuredLLM(),
        judge_llm=_DemoJudgeLLM(),
    )
    result = graph.run("What is the capital of France?")
    _report(result)

    print()
    print("== demo 2: exhausted retries → ErrorOutput ==")

    class _AlwaysLow:
        def generate(
            self, *, system: str, user: str, schema: type[BaseModel]
        ) -> BaseModel:
            return GroundedOutput(
                claims=[
                    Claim(
                        text="Unsure",
                        confidence=0.2,
                        sources=["https://example.com/x"],
                    )
                ]
            )

    bounded = Graph(
        domain=GeneralDomain(),
        structured_llm=_AlwaysLow(),
        judge_llm=_DemoJudgeLLM(),
        max_retries=2,
    )
    failed = bounded.run("Tell me something you do not know.")
    _report(failed)

    print()
    print("== demo 3: per-node progress via Graph.stream() ==")
    streaming_graph = Graph(
        domain=GeneralDomain(),
        structured_llm=_DemoStructuredLLM(),
        judge_llm=_DemoJudgeLLM(),
    )
    for event in streaming_graph.stream("What is the capital of France?"):
        s = event.state
        marker = "[ok] " if s.is_success else "[..] "
        print(
            f"  {marker}{event.node:<10} retry={s.retry_count} "
            f"gate={s.gate_result} critic={s.critic_result}"
        )

    print()
    print("== demo 4: per-node progress via Graph.astream() ==")
    async_graph = Graph(
        domain=GeneralDomain(),
        structured_llm=_DemoStructuredLLM(),
        judge_llm=_DemoJudgeLLM(),
    )

    async def _drain() -> None:
        async for event in async_graph.astream("What is the capital of France?"):
            s = event.state
            marker = "[ok] " if s.is_success else "[..] "
            print(
                f"  {marker}{event.node:<10} retry={s.retry_count} "
                f"gate={s.gate_result} critic={s.critic_result}"
            )

    asyncio.run(_drain())

    print()
    print("== demo 5: async-native AsyncStructuredLLM + AsyncJudgeLLM via Graph.arun() ==")
    async_native_graph = Graph(
        domain=GeneralDomain(),
        structured_llm=_AsyncDemoStructuredLLM(),
        judge_llm=_AsyncDemoJudgeLLM(),
    )

    async def _drive_async_native() -> GraphState:
        return await async_native_graph.arun("What is the capital of France?")

    _report(asyncio.run(_drive_async_native()))

    print()
    print("== demo 6: parallel research with two StructuredLLM clients ==")
    encyclopedist = _PersonaStructuredLLM(
        persona="encyclopedist",
        claim_text="Paris has been the capital of France since 987 AD",
        source="https://en.wikipedia.org/wiki/Paris",
    )
    geographer = _PersonaStructuredLLM(
        persona="geographer",
        claim_text="Paris sits on the Seine in the Île-de-France region",
        source="https://en.wikipedia.org/wiki/%C3%8Ele-de-France",
    )
    parallel_graph = Graph(
        domain=GeneralDomain(),
        structured_llm=[encyclopedist, geographer],
        judge_llm=_DemoJudgeLLM(),
    )
    print(
        f"  is_parallel    : {parallel_graph.is_parallel} "
        f"(num_researchers={parallel_graph.num_researchers})"
    )
    parallel_result = parallel_graph.run("Tell me about Paris.")
    _report(parallel_result)
    print(f"  branch_outputs : {len(parallel_result.branch_outputs)} entries")
    if parallel_result.research_output is not None:
        for claim in parallel_result.research_output.claims:
            print(f"    - {claim.text}")


if __name__ == "__main__":
    main()
