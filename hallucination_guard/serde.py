"""Serialization helpers for the LangGraph checkpointer integration.

LangGraph 1.1+ records this framework's Pydantic types (``GroundedOutput``,
``Claim``, ``CriticVerdict``, ``GraphState``) into checkpoints. When the
checkpointer reads them back it consults an msgpack allowlist; types
outside the allowlist trigger

    Deserializing unregistered type ... This will be blocked in a future
    version. Set LANGGRAPH_STRICT_MSGPACK=true to block now, or add to
    allowed_msgpack_modules to allow explicitly: [(...)]

today and a hard error once strict deserialization is enforced.

:func:`build_serializer` returns a :class:`JsonPlusSerializer` preloaded
with the framework's types so the warning disappears now and the code keeps
working when the allowlist becomes mandatory. Pass extra Pydantic classes
if you ship a custom :class:`hallucination_guard.schemas.GroundedOutput`
subclass via :meth:`DomainConfig.output_schema`.

Typical usage::

    from langgraph.checkpoint.memory import InMemorySaver
    from hallucination_guard.serde import build_serializer

    saver = InMemorySaver(serde=build_serializer())
    graph = Graph(..., checkpointer=saver)
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from hallucination_guard.schemas import Claim, CriticVerdict, GroundedOutput
from hallucination_guard.state import FailReason, GraphState


_FRAMEWORK_TYPES: tuple[type, ...] = (
    Claim,
    CriticVerdict,
    GroundedOutput,
    GraphState,
    FailReason,
)


def framework_msgpack_modules(
    *extra_types: type,
) -> tuple[tuple[str, str], ...]:
    """Return the ``(module, classname)`` allowlist for this framework's types.

    Suitable for passing directly to
    ``JsonPlusSerializer(allowed_msgpack_modules=...)`` if a caller needs
    to compose the allowlist with their own entries before constructing
    the serializer themselves.
    """
    all_types: tuple[type, ...] = _FRAMEWORK_TYPES + tuple(extra_types)
    return tuple((t.__module__, t.__name__) for t in all_types)


def build_serializer(*extra_types: type) -> JsonPlusSerializer:
    """Return a :class:`JsonPlusSerializer` allowing this framework's types.

    ``extra_types`` are merged into the allowlist so custom Pydantic models
    injected via :meth:`DomainConfig.output_schema` can also round-trip.
    """
    return JsonPlusSerializer(
        allowed_msgpack_modules=framework_msgpack_modules(*extra_types),
    )


def install_framework_serializer(checkpointer: Any, *extra_types: type) -> None:
    """Swap a *default* :class:`JsonPlusSerializer` on ``checkpointer`` for
    one allow-listing this framework's types.

    The replacement only happens when the checkpointer's ``serde`` is the
    LangGraph default — a :class:`JsonPlusSerializer` whose
    ``allowed_msgpack_modules`` is still the permissive ``True``. A
    customized serializer (different class, or narrowed allowlist) raises
    :class:`ValueError` instead of being overwritten; callers can then
    compose the allowlist explicitly via :func:`framework_msgpack_modules`.

    ``extra_types`` are forwarded to :func:`build_serializer`.
    """
    serde = getattr(checkpointer, "serde", None)
    if not isinstance(serde, JsonPlusSerializer):
        raise ValueError(
            "install_framework_serializer expected a checkpointer whose "
            "'serde' is a JsonPlusSerializer; got "
            f"{type(serde).__name__!r}. Build the serializer manually via "
            "hallucination_guard.serde.build_serializer()."
        )
    if getattr(serde, "_allowed_msgpack_modules", None) is not True:
        raise ValueError(
            "install_framework_serializer refuses to overwrite a customized "
            "JsonPlusSerializer (its allowed_msgpack_modules has already "
            "been narrowed). Merge the modules yourself via "
            "hallucination_guard.serde.framework_msgpack_modules()."
        )
    checkpointer.serde = build_serializer(*extra_types)


__all__ = [
    "build_serializer",
    "framework_msgpack_modules",
    "install_framework_serializer",
]
