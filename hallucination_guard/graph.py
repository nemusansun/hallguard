"""Graph — end-to-end pipeline wiring the nodes together with LangGraph.

The graph composes StructuredNode -> FactCheckGate -> CriticNode with retry
loops back through RetryNode and a terminal ErrorOutput branch reached when
the retry budget is exhausted.

Router functions check ``retry_count >= max_retries`` before any other
condition so the graph is provably bounded: after ``max_retries`` rejected
attempts, the next failed verdict routes to ``error`` instead of ``retry``.

LangGraph nodes return ``dict`` updates rather than full state instances;
``_wrap`` adapts the in-process ``GraphState``-returning node callables to
that contract while preserving their direct unit-test ergonomics. For
fields annotated with ``Annotated[list, operator.add]`` (reducer fields),
``_wrap`` returns only the **delta** (new items appended by the node) so
that LangGraph's built-in reducer concatenates rather than replaces.

Sync and async LLM clients can both be injected. The constructor inspects
the structured and judge clients against the runtime-checkable protocols
and, if either is async-only, compiles an async-wrapped graph. The async
graph is only drivable through :meth:`arun` and :meth:`astream`; the sync
:meth:`run` / :meth:`stream` entry points raise a clear error so callers
do not silently get coroutines back from LangGraph.

When ``structured_llm`` is a *list* of clients, the graph switches to
**parallel mode**: a dispatch node fans out via ``Send`` to N parallel
``StructuredNode`` invocations, an ``AggregatorNode`` merges their
outputs, and the rest of the pipeline runs as usual. Single-client
callers see no change.
"""

from __future__ import annotations

import operator as _operator
import typing as _typing
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from hallucination_guard.domain.base import DomainConfig
from hallucination_guard.llm.protocols import (
    AsyncJudgeLLM,
    AsyncStructuredLLM,
    JudgeLLM,
    StructuredLLM,
)
from hallucination_guard.nodes.aggregator import AggregatorNode
from hallucination_guard.nodes.critic_node import CriticNode
from hallucination_guard.nodes.error_output import ErrorOutput
from hallucination_guard.nodes.factcheck_gate import FactCheckGate
from hallucination_guard.nodes.retry_node import RetryNode
from hallucination_guard.nodes.structured_node import StructuredNode
from hallucination_guard.serde import install_framework_serializer
from hallucination_guard.state import GraphState


_STRUCTURED = "structured"
_FACTCHECK = "factcheck"
_CRITIC = "critic"
_RETRY = "retry"
_ERROR = "error"
_DISPATCH = "dispatch"
_AGGREGATE = "aggregate"


NodeCallable = Callable[[GraphState], GraphState]


def _find_additive_fields() -> frozenset[str]:
    """Detect ``GraphState`` fields annotated with ``operator.add``.

    These are the reducer fields whose values must be returned as deltas
    (new items only) by ``_wrap`` so LangGraph concatenates rather than
    replaces them when merging parallel updates.
    """
    hints = _typing.get_type_hints(GraphState, include_extras=True)
    found: set[str] = set()
    for name, hint in hints.items():
        if _typing.get_origin(hint) is _typing.Annotated:
            for meta in _typing.get_args(hint)[1:]:
                if meta is _operator.add:
                    found.add(name)
    return frozenset(found)


_ADDITIVE_FIELDS: frozenset[str] = _find_additive_fields()


@dataclass(frozen=True)
class StreamEvent:
    """One step of a streaming run: the node that just executed and the
    cumulative :class:`GraphState` after merging that node's update.

    Yielded by :meth:`Graph.stream` so callers can drive progress UIs or
    log per-node telemetry without losing access to the full state.
    """

    node: str
    state: GraphState


def _route_after_gate(state: GraphState) -> str:
    """Decide the next node after FactCheckGate.

    The ``retry_count >= max_retries`` guard is evaluated **before** the
    FAIL branch so an exhausted budget always terminates via ErrorOutput
    instead of looping back through RetryNode.
    """
    if state.gate_result == "FAIL":
        if state.retry_count >= state.max_retries:
            return _ERROR
        return _RETRY
    return _CRITIC


def _route_after_critic(state: GraphState) -> str:
    """Decide the next node after CriticNode."""
    if state.critic_result == "PASS":
        return END
    if state.retry_count >= state.max_retries:
        return _ERROR
    return _RETRY


def _coerce_state(state: GraphState | dict[str, Any]) -> GraphState:
    if isinstance(state, dict):
        return GraphState.model_validate(state)
    return state


def _build_update_dict(
    old: GraphState,
    new: GraphState,
    *,
    skip_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Build a LangGraph-compatible update dict from *old* -> *new*.

    For reducer fields (``_ADDITIVE_FIELDS``), only the **delta** (items
    appended beyond the original list) is emitted so LangGraph's
    ``operator.add`` concatenates correctly.  Empty deltas are omitted so
    parallel ``Send`` branches do not trigger "multiple values per step"
    errors on reducer channels.

    For non-reducer fields, only *changed* values are included.  Omitting
    unchanged fields is critical for ``Send``-based parallelism: LangGraph
    rejects concurrent writes of identical values to non-reducer channels.

    Fields listed in ``skip_fields`` are always excluded from the result.

    ``research_output`` is returned as a live attribute value (not via
    ``model_dump``) so Pydantic instances survive --- dumping would
    collapse them to ``dict`` and break duck-typing on ``.claims``.
    """
    result: dict[str, Any] = {}
    for name in GraphState.model_fields:
        if name in skip_fields:
            continue
        new_val = getattr(new, name)
        if name in _ADDITIVE_FIELDS:
            old_val = getattr(old, name)
            delta = new_val[len(old_val):]
            if delta:
                result[name] = delta
        else:
            old_val = getattr(old, name)
            if new_val is not old_val and new_val != old_val:
                result[name] = new_val
    return result


# Fields that Send-dispatched nodes must not write (handled by AggregatorNode).
_PARALLEL_SKIP: frozenset[str] = frozenset({"research_output"})


def _wrap(node: NodeCallable) -> Callable[[GraphState], dict[str, Any]]:
    """Adapt a ``GraphState -> GraphState`` node to LangGraph's dict-update contract."""

    def runner(state: GraphState) -> dict[str, Any]:
        coerced = _coerce_state(state)
        new_state = node(coerced)
        return _build_update_dict(coerced, new_state)

    return runner


def _wrap_parallel(node: NodeCallable) -> Callable[[GraphState], dict[str, Any]]:
    """Like ``_wrap`` but skips fields that conflict under Send parallelism."""

    def runner(state: GraphState) -> dict[str, Any]:
        coerced = _coerce_state(state)
        new_state = node(coerced)
        return _build_update_dict(coerced, new_state, skip_fields=_PARALLEL_SKIP)

    return runner


def _wrap_async(
    node: Any,
) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    """Async counterpart of :func:`_wrap`.

    Awaits :meth:`acall` when the wrapped node exposes one (the LLM-bound
    nodes do); otherwise calls the sync ``__call__`` directly. Pure-CPU
    nodes (FactCheckGate, RetryNode, ErrorOutput) have no async path, so
    the fallback keeps them usable inside an async-mode graph without
    forcing them to implement an empty ``acall``.
    """

    acall = getattr(node, "acall", None)

    async def runner(state: GraphState) -> dict[str, Any]:
        coerced = _coerce_state(state)
        if acall is not None:
            new_state = await acall(coerced)
        else:
            new_state = node(coerced)
        return _build_update_dict(coerced, new_state)

    return runner


def _wrap_async_parallel(
    node: Any,
) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    """Async counterpart of :func:`_wrap_parallel`."""

    acall = getattr(node, "acall", None)

    async def runner(state: GraphState) -> dict[str, Any]:
        coerced = _coerce_state(state)
        if acall is not None:
            new_state = await acall(coerced)
        else:
            new_state = node(coerced)
        return _build_update_dict(coerced, new_state, skip_fields=_PARALLEL_SKIP)

    return runner


def _is_async_client(client: Any, async_proto: type, sync_proto: type) -> bool:
    """Return ``True`` if ``client`` should be driven through the async API.

    A client that satisfies the async protocol but not the sync one is
    treated as async-only. When a class implements both surfaces, the sync
    path is preferred so existing :meth:`Graph.run` callers see no change
    in behavior — users who want async despite exposing both methods can
    drive the pipeline through :meth:`Graph.astream` explicitly.
    """
    return isinstance(client, async_proto) and not isinstance(client, sync_proto)


class Graph:
    """End-to-end pipeline configured for a single :class:`DomainConfig`.

    Inject the structured-output and judge LLMs explicitly so the framework
    stays free of any vendor-specific dependency; concrete adapters live
    outside this class.

    **Single-LLM mode** (default): pass one ``structured_llm`` and one
    ``judge_llm``. The graph runs the familiar serial pipeline.

    **Parallel mode**: pass a *list* of ``structured_llm`` clients. The
    graph fans out via LangGraph ``Send`` to N parallel ``StructuredNode``
    invocations, merges their outputs through an ``AggregatorNode``, then
    continues with the FactCheckGate / CriticNode pipeline. Each retry
    re-dispatches all N researchers.

    Each LLM slot accepts either a sync or async client; the constructor
    decides whether the graph runs through sync or async wrappers based on
    those types.

    A LangGraph ``checkpointer`` (e.g. ``InMemorySaver``) can be supplied to
    persist intermediate state. When set, callers must pass ``thread_id`` on
    :meth:`run` so LangGraph knows which conversation to associate the
    snapshot with; the persisted state can then be inspected through
    :meth:`get_state`.

    Set ``auto_serialize=True`` to opt into having the checkpointer's
    serializer swapped for one that allow-lists this framework's Pydantic
    types --- equivalent to constructing the checkpointer with
    ``serde=build_serializer()`` yourself. The swap is rejected when the
    checkpointer already carries a customized serializer so user-supplied
    allowlists are never silently clobbered.
    """

    def __init__(
        self,
        domain: DomainConfig,
        structured_llm: (
            StructuredLLM
            | AsyncStructuredLLM
            | list[StructuredLLM | AsyncStructuredLLM]
        ),
        judge_llm: JudgeLLM | AsyncJudgeLLM,
        max_retries: int = 3,
        checkpointer: Any = None,
        auto_serialize: bool = False,
        merge_strategy: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if auto_serialize:
            if checkpointer is None:
                raise ValueError(
                    "auto_serialize=True requires a checkpointer"
                )
            install_framework_serializer(checkpointer)
        self.domain = domain
        self.max_retries = max_retries
        self._checkpointer = checkpointer

        # Normalise structured_llm to a list for uniform handling.
        if isinstance(structured_llm, list):
            llm_list = structured_llm
            if not llm_list:
                raise ValueError("structured_llm list must not be empty")
        else:
            llm_list = [structured_llm]

        self._parallel = len(llm_list) > 1
        self._num_researchers = len(llm_list)

        # Detect async mode from any LLM in the list.
        any_struct_async = any(
            _is_async_client(s, AsyncStructuredLLM, StructuredLLM)
            for s in llm_list
        )
        judge_is_async = _is_async_client(judge_llm, AsyncJudgeLLM, JudgeLLM)
        self._async_mode = any_struct_async or judge_is_async

        common = dict(
            factcheck=FactCheckGate(domain),
            critic=CriticNode(domain, judge_llm),
            retry=RetryNode(),
            error=ErrorOutput(),
            checkpointer=checkpointer,
            async_mode=self._async_mode,
        )

        if self._parallel:
            self._compiled = self._build_parallel(
                domain=domain,
                llm_list=llm_list,
                num_researchers=self._num_researchers,
                merge_strategy=merge_strategy,
                **common,
            )
        else:
            self._compiled = self._build(
                structured=StructuredNode(domain, llm_list[0]),
                **common,
            )

    @property
    def is_async(self) -> bool:
        """Return ``True`` when at least one configured client is async-only.

        Inspected by sync entry points so they can refuse to drive an
        async-wrapped LangGraph (which would hand the caller raw coroutines
        instead of a state).
        """
        return self._async_mode

    @property
    def is_parallel(self) -> bool:
        """Return ``True`` when the graph was built with multiple researchers."""
        return self._parallel

    @property
    def num_researchers(self) -> int:
        """Number of parallel structured-LLM branches (1 when serial)."""
        return self._num_researchers

    def run(self, query: str, *, thread_id: str | None = None) -> GraphState:
        """Execute the graph for ``query`` and return the terminal state.

        The returned :class:`GraphState` exposes ``is_success``,
        ``final_output``, and ``error_message`` so callers can branch on
        outcome without catching exceptions.

        Pass ``thread_id`` when a checkpointer was supplied so LangGraph can
        persist state under that key.

        Raises :class:`RuntimeError` when the graph is configured with an
        async-only client --- use :meth:`arun` instead.
        """
        if self._async_mode:
            raise RuntimeError(
                "Graph is configured with an async LLM client; "
                "use arun() or astream() instead of run()."
            )
        self._require_thread_id(thread_id)

        initial = GraphState(user_query=query, max_retries=self.max_retries)
        config = self._make_config(thread_id)
        result: Any = self._compiled.invoke(initial, config=config)
        if isinstance(result, GraphState):
            return result
        return GraphState.model_validate(result)

    async def arun(
        self, query: str, *, thread_id: str | None = None
    ) -> GraphState:
        """Async single-shot counterpart of :meth:`run`.

        Works regardless of whether the configured clients are sync or
        async --- sync nodes are awaited through their dict-update wrapper
        so a uniform async surface is available to callers that already
        live inside an event loop.
        """
        self._require_thread_id(thread_id)

        initial = GraphState(user_query=query, max_retries=self.max_retries)
        config = self._make_config(thread_id)
        result: Any = await self._compiled.ainvoke(initial, config=config)
        if isinstance(result, GraphState):
            return result
        return GraphState.model_validate(result)

    def stream(
        self, query: str, *, thread_id: str | None = None
    ) -> Iterator[StreamEvent]:
        """Yield a :class:`StreamEvent` after each node executes.

        The accumulated :class:`GraphState` carried on each event is the
        result of merging successive node updates into the initial state,
        so the final yielded event's ``state`` equals what :meth:`run`
        would return for the same query.

        Pass ``thread_id`` when a checkpointer was supplied so LangGraph
        can persist state under that key, mirroring :meth:`run`.

        Raises :class:`RuntimeError` when the graph is configured with an
        async-only client --- use :meth:`astream` instead.
        """
        if self._async_mode:
            raise RuntimeError(
                "Graph is configured with an async LLM client; "
                "use astream() instead of stream()."
            )
        self._require_thread_id(thread_id)

        initial = GraphState(user_query=query, max_retries=self.max_retries)
        config = self._make_config(thread_id)

        accumulated = initial
        field_names = set(GraphState.model_fields)
        for chunk in self._compiled.stream(
            initial, config=config, stream_mode="updates"
        ):
            if not isinstance(chunk, dict):
                continue
            for node_name, update in chunk.items():
                merged = self._merge_update(accumulated, update, field_names)
                if merged is None:
                    continue
                accumulated = merged
                yield StreamEvent(node=node_name, state=accumulated)

    async def astream(
        self, query: str, *, thread_id: str | None = None
    ) -> AsyncIterator[StreamEvent]:
        """Asynchronous counterpart of :meth:`stream`.

        Yields the same :class:`StreamEvent` sequence as :meth:`stream` and
        applies the identical cumulative-update strategy that preserves
        Pydantic instances in ``research_output``; the difference is that
        node coroutines awaited by LangGraph can run without blocking the
        surrounding event loop, which matters once async LLM adapters enter
        the pipeline. Works regardless of whether the configured clients
        are sync or async.
        """
        self._require_thread_id(thread_id)

        initial = GraphState(user_query=query, max_retries=self.max_retries)
        config = self._make_config(thread_id)

        accumulated = initial
        field_names = set(GraphState.model_fields)
        async for chunk in self._compiled.astream(
            initial, config=config, stream_mode="updates"
        ):
            if not isinstance(chunk, dict):
                continue
            for node_name, update in chunk.items():
                merged = self._merge_update(accumulated, update, field_names)
                if merged is None:
                    continue
                accumulated = merged
                yield StreamEvent(node=node_name, state=accumulated)

    def get_state(self, thread_id: str) -> GraphState:
        """Return the persisted state for ``thread_id``.

        Requires a checkpointer; raises :class:`RuntimeError` otherwise.
        """
        if self._checkpointer is None:
            raise RuntimeError("get_state requires a checkpointer")
        snapshot: Any = self._compiled.get_state(
            {"configurable": {"thread_id": thread_id}}
        )
        values: Any = snapshot.values
        if isinstance(values, GraphState):
            return values
        return GraphState.model_validate(values)

    def _require_thread_id(self, thread_id: str | None) -> None:
        if self._checkpointer is not None and thread_id is None:
            raise ValueError(
                "thread_id is required when a checkpointer is configured"
            )

    @staticmethod
    def _make_config(thread_id: str | None) -> dict[str, Any] | None:
        if thread_id is None:
            return None
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _merge_update(
        accumulated: GraphState, update: Any, field_names: set[str]
    ) -> GraphState | None:
        if isinstance(update, GraphState):
            return update
        if isinstance(update, dict):
            filtered = {k: v for k, v in update.items() if k in field_names}
            # Manually apply additive reducers: stream_mode="updates"
            # returns the raw node delta (before LangGraph's reducer),
            # so we must concatenate it with the accumulated state.
            for name in _ADDITIVE_FIELDS:
                if name in filtered:
                    old = getattr(accumulated, name)
                    filtered[name] = old + filtered[name]
            return accumulated.with_update(**filtered)
        return None

    @staticmethod
    def _build(
        *,
        structured: Any,
        factcheck: Any,
        critic: Any,
        retry: Any,
        error: Any,
        checkpointer: Any = None,
        async_mode: bool = False,
    ) -> Any:
        wrap = _wrap_async if async_mode else _wrap
        graph: Any = StateGraph(GraphState)
        graph.add_node(_STRUCTURED, wrap(structured))
        graph.add_node(_FACTCHECK, wrap(factcheck))
        graph.add_node(_CRITIC, wrap(critic))
        graph.add_node(_RETRY, wrap(retry))
        graph.add_node(_ERROR, wrap(error))

        graph.add_edge(START, _STRUCTURED)
        graph.add_edge(_STRUCTURED, _FACTCHECK)
        graph.add_conditional_edges(
            _FACTCHECK,
            _route_after_gate,
            {_RETRY: _RETRY, _CRITIC: _CRITIC, _ERROR: _ERROR},
        )
        graph.add_conditional_edges(
            _CRITIC,
            _route_after_critic,
            {_RETRY: _RETRY, _ERROR: _ERROR, END: END},
        )
        graph.add_edge(_RETRY, _STRUCTURED)
        graph.add_edge(_ERROR, END)

        return graph.compile(checkpointer=checkpointer)

    @staticmethod
    def _build_parallel(
        *,
        domain: DomainConfig,
        llm_list: list[Any],
        num_researchers: int,
        merge_strategy: Callable[[list[Any]], Any] | None,
        factcheck: Any,
        critic: Any,
        retry: Any,
        error: Any,
        checkpointer: Any = None,
        async_mode: bool = False,
    ) -> Any:
        """Build a fan-out / fan-in graph for parallel research."""
        wrap = _wrap_async if async_mode else _wrap
        wrap_par = _wrap_async_parallel if async_mode else _wrap_parallel
        parallel_node = _ParallelStructuredNode(domain, llm_list)
        aggregator = AggregatorNode(num_researchers, merge_strategy)

        def _pass_through(state: GraphState) -> GraphState:
            return state

        def _fan_out(state: GraphState) -> list[Send]:
            base = {
                name: getattr(state, name) for name in GraphState.model_fields
            }
            return [
                Send(_STRUCTURED, {**base, "researcher_id": i})
                for i in range(num_researchers)
            ]

        graph: Any = StateGraph(GraphState)
        graph.add_node(_DISPATCH, wrap(_pass_through))
        graph.add_node(_STRUCTURED, wrap_par(parallel_node))
        graph.add_node(_AGGREGATE, wrap(aggregator))
        graph.add_node(_FACTCHECK, wrap(factcheck))
        graph.add_node(_CRITIC, wrap(critic))
        graph.add_node(_RETRY, wrap(retry))
        graph.add_node(_ERROR, wrap(error))

        graph.add_edge(START, _DISPATCH)
        graph.add_conditional_edges(_DISPATCH, _fan_out, [_STRUCTURED])
        graph.add_edge(_STRUCTURED, _AGGREGATE)
        graph.add_edge(_AGGREGATE, _FACTCHECK)
        graph.add_conditional_edges(
            _FACTCHECK,
            _route_after_gate,
            {_RETRY: _RETRY, _CRITIC: _CRITIC, _ERROR: _ERROR},
        )
        graph.add_conditional_edges(
            _CRITIC,
            _route_after_critic,
            {_RETRY: _RETRY, _ERROR: _ERROR, END: END},
        )
        graph.add_edge(_RETRY, _DISPATCH)
        graph.add_edge(_ERROR, END)

        return graph.compile(checkpointer=checkpointer)


class _ParallelStructuredNode:
    """Dispatches to one of N ``StructuredNode`` instances based on ``researcher_id``.

    Used internally by the parallel graph; each ``Send`` sets a different
    ``researcher_id`` in the state so the correct LLM is selected.
    """

    def __init__(
        self,
        domain: DomainConfig,
        llms: list[StructuredLLM | AsyncStructuredLLM],
    ) -> None:
        self._nodes = [StructuredNode(domain, llm) for llm in llms]

    def __call__(self, state: GraphState) -> GraphState:
        return self._nodes[state.researcher_id](state)

    async def acall(self, state: GraphState) -> GraphState:
        return await self._nodes[state.researcher_id].acall(state)
