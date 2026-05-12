"""FactCheckGate — confidence threshold and source-validity gate.

Reads ``state.research_output`` (expected to expose a ``.claims`` list) and
decides whether the output may proceed to :class:`CriticNode`.

Failure ordering: confidence is checked **before** source validity. When
multiple claims fail the same check, one ``fail_history`` entry is appended
per failing claim, all under the same ``fail_reason``.
"""

from __future__ import annotations

from typing import Any

from hallucination_guard.domain.base import DomainConfig
from hallucination_guard.exceptions import GraphError
from hallucination_guard.state import FailReason, GraphState


class FactCheckGate:
    """Confidence and source-validity check."""

    def __init__(self, domain: DomainConfig) -> None:
        self.domain = domain

    def __call__(self, state: GraphState) -> GraphState:
        output = state.research_output
        if output is None:
            raise GraphError("FactCheckGate received empty research_output")

        claims = self._extract_claims(output)

        low_conf = [
            c for c in claims if c.confidence < self.domain.confidence_threshold
        ]
        if low_conf:
            additions = [
                f"{FailReason.LOW_CONFIDENCE.value}:{c.text}" for c in low_conf
            ]
            return state.with_update(
                gate_result="FAIL",
                fail_reason=FailReason.LOW_CONFIDENCE,
                fail_history=state.fail_history + additions,
            )

        no_source = [
            c
            for c in claims
            if not c.sources
            or not any(self.domain.is_valid_source(s) for s in c.sources)
        ]
        if no_source:
            additions = [
                f"{FailReason.NO_SOURCE.value}:{c.text}" for c in no_source
            ]
            return state.with_update(
                gate_result="FAIL",
                fail_reason=FailReason.NO_SOURCE,
                fail_history=state.fail_history + additions,
            )

        return state.with_update(gate_result="PASS")

    @staticmethod
    def _extract_claims(output: Any) -> list[Any]:
        if not hasattr(output, "claims"):
            raise GraphError(
                "research_output must expose a `.claims` attribute; "
                f"got {type(output).__name__}"
            )
        return list(output.claims)
