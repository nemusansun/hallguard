"""RetryHintBuilder — converts a ``GraphState`` into a safe ``RetryDirective``.

The builder is the single chokepoint between the graph's failure history and
the next prompt. It asks the active :class:`DomainConfig` for the
fix-instruction phrase keyed to ``state.fail_reason`` — wording (including
language) lives entirely in the domain. Raw ``fail_history`` content is
never forwarded; only claims extracted via
:meth:`GraphState.get_rejected_claims` populate ``forbidden_claims``.
"""

from __future__ import annotations

from hallucination_guard.domain.base import DomainConfig
from hallucination_guard.retry.directive import RetryDirective
from hallucination_guard.state import GraphState


class RetryHintBuilder:
    """Build a :class:`RetryDirective` from a state's ``fail_reason``."""

    @classmethod
    def build(cls, state: GraphState, domain: DomainConfig) -> RetryDirective:
        """Return a :class:`RetryDirective` derived from ``state``.

        Args:
            state: The current graph state. ``state.fail_reason`` must not be
                ``None``.
            domain: The active domain configuration. Its
                :meth:`DomainConfig.retry_instruction` supplies the
                ``fix_instruction`` phrase.

        Raises:
            ValueError: If ``state.fail_reason`` is ``None`` — there is no
                signal from which to derive a fix instruction.

        Notes:
            ``state.fail_history`` is never forwarded verbatim. Only claims
            extracted via :meth:`GraphState.get_rejected_claims` (a structured
            view) populate ``forbidden_claims``.
        """
        if state.fail_reason is None:
            raise ValueError(
                "Cannot build a RetryDirective: state.fail_reason is None"
            )

        fix_instruction = domain.retry_instruction(state.fail_reason)
        forbidden_claims = state.get_rejected_claims()

        return RetryDirective(
            fix_instruction=fix_instruction,
            forbidden_claims=forbidden_claims,
        )
