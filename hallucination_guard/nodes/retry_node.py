"""RetryNode — prepares state for the next StructuredNode attempt.

Runs after a failed gate or critic verdict. Increments ``retry_count`` and
clears the stale ``gate_result`` / ``critic_result`` signals so the next
iteration can write fresh values. The hint itself is built on demand inside
:class:`StructuredNode` from ``state.fail_reason`` and ``state.fail_history``;
RetryNode does not carry the hint in state.
"""

from __future__ import annotations

from hallucination_guard.state import GraphState


class RetryNode:
    """Bump the retry counter and clear stale per-attempt signals."""

    def __call__(self, state: GraphState) -> GraphState:
        return state.with_update(
            retry_count=state.retry_count + 1,
            gate_result=None,
            critic_result=None,
        )
