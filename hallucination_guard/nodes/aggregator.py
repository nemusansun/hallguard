"""AggregatorNode — merges parallel research outputs into a single result.

Used in parallel mode after all ``Send``-dispatched ``StructuredNode``
branches complete. Reads the last *N* entries from ``branch_outputs``
(where *N* is the number of researchers) and delegates merging to a
pluggable ``merge_fn``. The default strategy concatenates all ``.claims``
from every branch into a single :class:`GroundedOutput`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from hallucination_guard.schemas import GroundedOutput
from hallucination_guard.state import GraphState


def default_merge(outputs: list[Any]) -> Any:
    """Concatenate claims from all parallel outputs into one GroundedOutput."""
    all_claims: list[Any] = []
    for output in outputs:
        if hasattr(output, "claims"):
            all_claims.extend(output.claims)
    return GroundedOutput(claims=all_claims)


class AggregatorNode:
    """Merge ``branch_outputs`` from parallel structured nodes.

    Takes the last ``num_researchers`` items from ``branch_outputs`` (which
    accumulates across retries via its ``operator.add`` reducer) so that
    only the current round's outputs are merged.
    """

    def __init__(
        self,
        num_researchers: int,
        merge_fn: Optional[Callable[[list[Any]], Any]] = None,
    ) -> None:
        if num_researchers < 1:
            raise ValueError("num_researchers must be >= 1")
        self.num_researchers = num_researchers
        self.merge_fn = merge_fn or default_merge

    def __call__(self, state: GraphState) -> GraphState:
        latest = state.branch_outputs[-self.num_researchers:]
        merged = self.merge_fn(latest)
        return state.with_update(research_output=merged)
