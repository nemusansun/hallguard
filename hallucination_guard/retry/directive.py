"""RetryDirective — the only type allowed to carry retry hints into a prompt.

The directive is intentionally narrow: a fixed ``fix_instruction`` string plus
a list of ``forbidden_claims``. Raw ``fail_history`` text never reaches a
prompt.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetryDirective(BaseModel):
    """Immutable instructions injected into the next generation attempt.

    Attributes:
        fix_instruction: A fixed, hand-authored sentence describing what the
            next attempt must do differently. Sourced only from
            :meth:`DomainConfig.retry_instruction`.
        forbidden_claims: Claims previously rejected by the CriticNode that
            must not be repeated.
    """

    model_config = ConfigDict(frozen=True)

    fix_instruction: str
    forbidden_claims: list[str] = Field(default_factory=list)
