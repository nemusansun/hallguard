"""GraphState — the single immutable state object that flows through the graph.

State updates MUST go through :meth:`GraphState.with_update`, which delegates to
``model_copy(update=...)`` and returns a fresh instance. Direct mutation is
forbidden by convention.

``fail_history`` entries follow the convention ``"<FailReason.value>:<detail>"``
so that :meth:`GraphState.get_rejected_claims` can pull out only the claims that
were rejected by the CriticNode.

List fields annotated with ``Annotated[list, operator.add]`` are **reducer
fields**: when LangGraph merges updates from parallel nodes it concatenates
the returned delta with the existing list instead of replacing it. Node
callables may continue to use the ``state.field + additions`` concat pattern
internally — the ``_wrap`` adapter in ``graph.py`` computes the delta
automatically before handing the dict to LangGraph.
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field


class FailReason(str, Enum):
    """Why a generation was rejected by the gate or the critic."""

    LOW_CONFIDENCE = "low_confidence"
    NO_SOURCE = "no_source"
    CRITIC_REJECTED = "critic_rejected"


class GraphState(BaseModel):
    """Immutable state passed between nodes.

    Update via :meth:`with_update` to obtain a new instance; never mutate
    fields in place.
    """

    user_query: str
    research_output: Optional[Any] = None
    branch_outputs: Annotated[list[Any], operator.add] = Field(
        default_factory=list,
    )
    researcher_id: int = 0
    retry_count: int = 0
    max_retries: int = 3
    fail_reason: Optional[FailReason] = None
    fail_history: Annotated[list[str], operator.add] = Field(
        default_factory=list,
    )
    gate_result: Optional[Literal["PASS", "FAIL"]] = None
    critic_result: Optional[Literal["PASS", "FAIL"]] = None
    is_success: bool = False
    final_output: Optional[str] = None
    error_message: Optional[str] = None

    def with_update(self, **kwargs: Any) -> GraphState:
        """Return a new ``GraphState`` with the given fields replaced.

        The original instance is left untouched. Internally this delegates to
        Pydantic's ``model_copy(update=...)``.
        """
        return self.model_copy(update=kwargs)

    def get_rejected_claims(self) -> list[str]:
        """Return claims that the CriticNode rejected in previous attempts.

        ``fail_history`` entries are expected to follow the convention
        ``"<FailReason.value>:<claim>"``. Only entries prefixed with
        :attr:`FailReason.CRITIC_REJECTED` are returned, with the prefix
        stripped and surrounding whitespace removed.
        """
        prefix = f"{FailReason.CRITIC_REJECTED.value}:"
        return [
            entry[len(prefix):].strip()
            for entry in self.fail_history
            if entry.startswith(prefix)
        ]
