"""RetryHintBuilder — converts a ``GraphState`` into a safe ``RetryDirective``.

The builder is the single chokepoint between the graph's failure history and
the next prompt. It MUST only emit fix-instruction strings from
:attr:`RetryHintBuilder.INSTRUCTION_MAPS`; raw ``fail_history`` content is
never forwarded.

Instruction phrases are keyed by :data:`Locale` so the language of the retry
prompt can match the language of the rest of the domain's prompts.
"""

from __future__ import annotations

from hallucination_guard.domain.base import Locale
from hallucination_guard.retry.directive import RetryDirective
from hallucination_guard.state import FailReason, GraphState


class RetryHintBuilder:
    """Build a :class:`RetryDirective` from a state's ``fail_reason``."""

    INSTRUCTION_MAPS: dict[Locale, dict[FailReason, str]] = {
        "en": {
            FailReason.LOW_CONFIDENCE: (
                "State each claim's confidence as a value in [0.0, 1.0]."
            ),
            FailReason.NO_SOURCE: (
                "Attach at least one source URL to every claim."
            ),
            FailReason.CRITIC_REJECTED: (
                "Do not repeat any claim that was rejected in a previous attempt."
            ),
        },
        "ja": {
            FailReason.LOW_CONFIDENCE: "各主張の確信度を0.0〜1.0で明示してください",
            FailReason.NO_SOURCE: "主張ごとに出典URLを必ず添付してください",
            FailReason.CRITIC_REJECTED: "前回否定された主張を繰り返さないでください",
        },
    }

    @classmethod
    def build(
        cls, state: GraphState, *, locale: Locale = "en"
    ) -> RetryDirective:
        """Return a :class:`RetryDirective` derived from ``state``.

        Args:
            state: The current graph state. ``state.fail_reason`` must not be
                ``None``.
            locale: Language for the ``fix_instruction`` phrase. Defaults to
                ``"en"``; pass ``"ja"`` (typically via
                :meth:`DomainConfig.retry_locale`) to surface Japanese
                instructions.

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

        fix_instruction = cls.INSTRUCTION_MAPS[locale][state.fail_reason]
        forbidden_claims = state.get_rejected_claims()

        return RetryDirective(
            fix_instruction=fix_instruction,
            forbidden_claims=forbidden_claims,
        )
