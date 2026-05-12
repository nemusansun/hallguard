"""Tests for :mod:`hallucination_guard.llm.openai_adapter`.

A small in-memory client mimics the shape of ``OpenAI().chat.completions.parse``
so the adapter can be exercised without any network call or API key.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest
from pydantic import BaseModel

from hallucination_guard.exceptions import CriticError, StructuredOutputError
from hallucination_guard.llm.openai_adapter import (
    OpenAIJudgeAdapter,
    OpenAIStructuredAdapter,
)
from hallucination_guard.schemas import Claim, CriticVerdict, GroundedOutput


_FAKE_MODEL = "test-model-a"
_FAKE_MODEL_ALT = "test-model-b"


class _FakeParse:
    """Records ``parse`` calls and returns a scripted parsed payload."""

    def __init__(self, parsed: Optional[BaseModel]) -> None:
        self._parsed = parsed
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=self._parsed)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def _client_with(parse_fn: _FakeParse) -> SimpleNamespace:
    """Wrap ``parse_fn`` in the nested attribute path the adapter expects."""
    completions = SimpleNamespace(parse=parse_fn)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def test_structured_adapter_returns_parsed_payload() -> None:
    payload = GroundedOutput(
        claims=[Claim(text="x", confidence=0.9, sources=["https://e.com"])]
    )
    parse = _FakeParse(parsed=payload)
    adapter = OpenAIStructuredAdapter(
        client=_client_with(parse), model=_FAKE_MODEL
    )

    result = adapter.generate(system="sys", user="usr", schema=GroundedOutput)

    assert result is payload


def test_structured_adapter_forwards_messages_and_schema() -> None:
    parse = _FakeParse(parsed=GroundedOutput())
    adapter = OpenAIStructuredAdapter(
        client=_client_with(parse), model=_FAKE_MODEL
    )

    adapter.generate(system="SYS", user="USR", schema=GroundedOutput)

    call = parse.calls[0]
    assert call["model"] == _FAKE_MODEL
    assert call["response_format"] is GroundedOutput
    assert call["temperature"] == 0.0
    assert call["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]


def test_structured_adapter_honors_custom_model_and_temperature() -> None:
    parse = _FakeParse(parsed=GroundedOutput())
    adapter = OpenAIStructuredAdapter(
        client=_client_with(parse),
        model=_FAKE_MODEL_ALT,
        temperature=0.3,
    )
    adapter.generate(system="s", user="u", schema=GroundedOutput)
    call = parse.calls[0]
    assert call["model"] == _FAKE_MODEL_ALT
    assert call["temperature"] == 0.3


def test_structured_adapter_raises_when_payload_is_none() -> None:
    parse = _FakeParse(parsed=None)
    adapter = OpenAIStructuredAdapter(
        client=_client_with(parse), model=_FAKE_MODEL
    )

    with pytest.raises(StructuredOutputError):
        adapter.generate(system="s", user="u", schema=GroundedOutput)


def test_structured_adapter_raises_on_unexpected_payload_type() -> None:
    class _Other(BaseModel):
        pass

    parse = _FakeParse(parsed=_Other())
    adapter = OpenAIStructuredAdapter(
        client=_client_with(parse), model=_FAKE_MODEL
    )

    with pytest.raises(StructuredOutputError):
        adapter.generate(system="s", user="u", schema=GroundedOutput)


def test_judge_adapter_returns_verdict() -> None:
    verdict = CriticVerdict(verdict="PASS")
    parse = _FakeParse(parsed=verdict)
    adapter = OpenAIJudgeAdapter(client=_client_with(parse), model=_FAKE_MODEL)

    result = adapter.judge(system="sys", content="content")

    assert result is verdict


def test_judge_adapter_binds_critic_verdict_schema() -> None:
    parse = _FakeParse(parsed=CriticVerdict(verdict="PASS"))
    adapter = OpenAIJudgeAdapter(client=_client_with(parse), model=_FAKE_MODEL)

    adapter.judge(system="sys", content="payload")

    call = parse.calls[0]
    assert call["response_format"] is CriticVerdict
    assert call["temperature"] == 0.0
    assert call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "payload"},
    ]


def test_judge_adapter_raises_when_payload_is_none() -> None:
    parse = _FakeParse(parsed=None)
    adapter = OpenAIJudgeAdapter(client=_client_with(parse), model=_FAKE_MODEL)

    with pytest.raises(CriticError):
        adapter.judge(system="s", content="c")


def test_judge_adapter_raises_on_unexpected_payload_type() -> None:
    parse = _FakeParse(parsed=GroundedOutput())  # wrong type
    adapter = OpenAIJudgeAdapter(client=_client_with(parse), model=_FAKE_MODEL)

    with pytest.raises(CriticError):
        adapter.judge(system="s", content="c")
