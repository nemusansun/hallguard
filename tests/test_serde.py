"""Tests for :mod:`hallucination_guard.serde`.

Verifies that

- the allowlist tuple includes every framework Pydantic type the graph
  routes through checkpoints
- extra types passed by the caller are merged in
- using the helper with an :class:`InMemorySaver` silences the
  ``Deserializing unregistered type`` warning emitted by LangGraph 1.1
  when the framework types appear in a persisted state
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from pydantic import BaseModel

from hallucination_guard.domain.general import GeneralDomain
from hallucination_guard.graph import Graph
from hallucination_guard.schemas import Claim, CriticVerdict, GroundedOutput
from hallucination_guard.serde import (
    build_serializer,
    framework_msgpack_modules,
    install_framework_serializer,
)
from hallucination_guard.state import FailReason, GraphState


def test_framework_modules_cover_persisted_types() -> None:
    modules = framework_msgpack_modules()
    names = {name for _, name in modules}
    assert {"Claim", "CriticVerdict", "GroundedOutput", "GraphState", "FailReason"} <= names


def test_framework_modules_use_concrete_import_paths() -> None:
    modules = framework_msgpack_modules()
    assert ("hallucination_guard.schemas", "GroundedOutput") in modules
    assert ("hallucination_guard.state", "GraphState") in modules
    assert ("hallucination_guard.state", "FailReason") in modules


def test_extra_types_are_merged() -> None:
    class CustomGroundedOutput(GroundedOutput):
        pass

    modules = framework_msgpack_modules(CustomGroundedOutput)
    assert ("tests.test_serde", "CustomGroundedOutput") in modules or any(
        name == "CustomGroundedOutput" for _, name in modules
    )


def test_build_serializer_returns_jsonplus_with_explicit_allowlist() -> None:
    serde = build_serializer()
    # Allowlist must not be the permissive True sentinel — that would
    # defeat the purpose of suppressing the runtime warning.
    assert serde._allowed_msgpack_modules is not True
    assert serde._allowed_msgpack_modules is not None


class _PassingStructured:
    def generate(self, *, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        return GroundedOutput(
            claims=[
                Claim(
                    text="Capital city",
                    confidence=0.97,
                    sources=["https://en.wikipedia.org/wiki/Example"],
                )
            ]
        )


class _PassingJudge:
    def judge(self, *, system: str, content: str) -> CriticVerdict:
        return CriticVerdict(verdict="PASS")


def test_checkpointer_with_helper_emits_no_unregistered_warning() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=_PassingStructured(),
        judge_llm=_PassingJudge(),
        checkpointer=InMemorySaver(serde=build_serializer()),
    )
    graph.run("query", thread_id="t1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        snapshot = graph.get_state("t1")

    leaked = [
        str(w.message)
        for w in caught
        if "unregistered type" in str(w.message).lower()
        or "allowed_msgpack" in str(w.message).lower()
    ]
    assert leaked == [], f"unexpected serialization warnings: {leaked}"
    assert snapshot.is_success is True
    assert snapshot.research_output is not None
    assert hasattr(snapshot.research_output, "claims")


def test_install_framework_serializer_replaces_default_serde() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    # Sanity check: vanilla InMemorySaver carries the permissive default.
    assert saver.serde._allowed_msgpack_modules is True

    install_framework_serializer(saver)
    assert saver.serde._allowed_msgpack_modules is not True
    assert ("hallucination_guard.schemas", "GroundedOutput") in (
        saver.serde._allowed_msgpack_modules
    )


def test_install_framework_serializer_refuses_customized_serde() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    custom = build_serializer()  # already narrowed
    saver = InMemorySaver(serde=custom)

    with pytest.raises(ValueError, match="customized"):
        install_framework_serializer(saver)
    # The original serde must remain in place.
    assert saver.serde is custom


def test_install_framework_serializer_rejects_non_jsonplus_serde() -> None:
    class _DummyCheckpointer:
        def __init__(self) -> None:
            self.serde = object()  # not a JsonPlusSerializer

    with pytest.raises(ValueError, match="JsonPlusSerializer"):
        install_framework_serializer(_DummyCheckpointer())


def test_graph_auto_serialize_swaps_default_checkpointer_serde() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    Graph(
        domain=GeneralDomain(),
        structured_llm=_PassingStructured(),
        judge_llm=_PassingJudge(),
        checkpointer=saver,
        auto_serialize=True,
    )
    # __init__ should have replaced the default permissive serde in place.
    assert saver.serde._allowed_msgpack_modules is not True


def test_graph_auto_serialize_requires_checkpointer() -> None:
    with pytest.raises(ValueError, match="checkpointer"):
        Graph(
            domain=GeneralDomain(),
            structured_llm=_PassingStructured(),
            judge_llm=_PassingJudge(),
            auto_serialize=True,
        )


def test_graph_auto_serialize_refuses_custom_serde() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver(serde=build_serializer())
    with pytest.raises(ValueError, match="customized"):
        Graph(
            domain=GeneralDomain(),
            structured_llm=_PassingStructured(),
            judge_llm=_PassingJudge(),
            checkpointer=saver,
            auto_serialize=True,
        )
