"""OpenAI implementations of the LLM client protocols.

Both sync (``OpenAIStructuredAdapter`` / ``OpenAIJudgeAdapter``) and async
(``AsyncOpenAIStructuredAdapter`` / ``AsyncOpenAIJudgeAdapter``) flavours are
provided. They all use ``chat.completions.parse`` with
``response_format=<Pydantic class>`` so the returned object is a validated
instance of the requested schema. The default temperature is ``0`` to match
the framework's determinism stance; callers can override per instance if a
strict-zero temperature is rejected by their chosen model.

The OpenAI client is injectable so tests can swap in an in-memory fake. The
default ``OpenAI()`` / ``AsyncOpenAI()`` instances read ``OPENAI_API_KEY``
from the environment.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from hallucination_guard.exceptions import CriticError, StructuredOutputError
from hallucination_guard.schemas import CriticVerdict


def _default_client() -> Any:
    """Build a default ``OpenAI`` client lazily so the import is optional.

    The return type is left as ``Any`` because the surface used by these
    adapters — ``client.chat.completions.parse(...)`` — is a duck-typed
    contract; tests inject in-memory stand-ins built from
    :class:`types.SimpleNamespace`.
    """
    from openai import OpenAI  # imported here to keep the dependency optional

    return OpenAI()


def _default_async_client() -> Any:
    """Build a default ``AsyncOpenAI`` client lazily.

    Mirrors :func:`_default_client` but for the async surface so adapter
    construction does not require the SDK to be importable at module load.
    """
    from openai import AsyncOpenAI  # imported here to keep the dependency optional

    return AsyncOpenAI()


def _validate_parsed(parsed: Any, expected: type[BaseModel], error_cls: type[Exception], *, model: str) -> Any:
    if parsed is None:
        raise error_cls(
            f"OpenAI returned no parsed payload "
            f"(model={model!r}, schema={expected.__name__!r})"
        )
    if not isinstance(parsed, expected):
        raise error_cls(
            f"OpenAI returned an unexpected payload type: "
            f"{type(parsed).__name__} (expected {expected.__name__})"
        )
    return parsed


class OpenAIStructuredAdapter:
    """Implements :class:`~hallucination_guard.llm.protocols.StructuredLLM`.

    The schema passed at call time is forwarded to OpenAI's structured-output
    endpoint, so the returned value is already a parsed instance of that
    schema. Tests should inject ``client`` directly.
    """

    def __init__(
        self,
        *,
        model: str,
        client: Any = None,
        temperature: float = 0.0,
    ) -> None:
        self._client = client if client is not None else _default_client()
        self._model = model
        self._temperature = temperature

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        completion = self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
            temperature=self._temperature,
        )
        parsed = completion.choices[0].message.parsed
        return _validate_parsed(
            parsed, schema, StructuredOutputError, model=self._model
        )


class OpenAIJudgeAdapter:
    """Implements :class:`~hallucination_guard.llm.protocols.JudgeLLM`.

    Always asks for a :class:`CriticVerdict`. The system prompt is supplied
    per call (typically from :meth:`DomainConfig.critic_prompt`).
    """

    def __init__(
        self,
        *,
        model: str,
        client: Any = None,
        temperature: float = 0.0,
    ) -> None:
        self._client = client if client is not None else _default_client()
        self._model = model
        self._temperature = temperature

    def judge(self, *, system: str, content: str) -> CriticVerdict:
        completion = self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            response_format=CriticVerdict,
            temperature=self._temperature,
        )
        parsed = completion.choices[0].message.parsed
        return _validate_parsed(
            parsed, CriticVerdict, CriticError, model=self._model
        )


class AsyncOpenAIStructuredAdapter:
    """Implements :class:`~hallucination_guard.llm.protocols.AsyncStructuredLLM`.

    Built around ``AsyncOpenAI`` so calls can be awaited inside an event
    loop without blocking. Injection-friendly for the same reason the sync
    adapter is: tests pass an in-memory client whose ``chat.completions.parse``
    is a coroutine returning a pre-baked completion.
    """

    def __init__(
        self,
        *,
        model: str,
        client: Any = None,
        temperature: float = 0.0,
    ) -> None:
        self._client = (
            client if client is not None else _default_async_client()
        )
        self._model = model
        self._temperature = temperature

    async def agenerate(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        completion = await self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
            temperature=self._temperature,
        )
        parsed = completion.choices[0].message.parsed
        return _validate_parsed(
            parsed, schema, StructuredOutputError, model=self._model
        )


class AsyncOpenAIJudgeAdapter:
    """Implements :class:`~hallucination_guard.llm.protocols.AsyncJudgeLLM`.

    Async counterpart of :class:`OpenAIJudgeAdapter`. Always asks for a
    :class:`CriticVerdict`.
    """

    def __init__(
        self,
        *,
        model: str,
        client: Any = None,
        temperature: float = 0.0,
    ) -> None:
        self._client = (
            client if client is not None else _default_async_client()
        )
        self._model = model
        self._temperature = temperature

    async def ajudge(self, *, system: str, content: str) -> CriticVerdict:
        completion = await self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            response_format=CriticVerdict,
            temperature=self._temperature,
        )
        parsed = completion.choices[0].message.parsed
        return _validate_parsed(
            parsed, CriticVerdict, CriticError, model=self._model
        )
