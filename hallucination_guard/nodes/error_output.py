"""ErrorOutput — terminal node reached when retries are exhausted.

Reached when the router observes ``state.retry_count >= state.max_retries``.
Produces a structured failure record on the state rather than raising, so
the graph completes cleanly and the caller can inspect ``is_success`` and
``error_message``.
"""

from __future__ import annotations

from hallucination_guard.state import GraphState


class ErrorOutput:
    """Mark the run as failed and emit a human-readable error message."""

    def __call__(self, state: GraphState) -> GraphState:
        last_reason = state.fail_reason.value if state.fail_reason else "unknown"
        message = (
            f"max_retries ({state.max_retries}) reached. "
            f"last_fail_reason={last_reason}, "
            f"total_failures={len(state.fail_history)}"
        )
        return state.with_update(
            is_success=False,
            final_output=None,
            error_message=message,
        )
