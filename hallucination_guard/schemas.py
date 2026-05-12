"""Shared data shapes used by nodes and domain configurations.

These types form the contract that lets a node grade output produced under
any domain: every domain's ``output_schema()`` is expected to expose a
``claims`` list of :class:`Claim`-shaped objects (typically by subclassing
:class:`GroundedOutput`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A single atomic assertion with a confidence score and citations.

    Attributes:
        text: The claim itself, in natural language.
        confidence: Self-reported probability that the claim is true (0.0–1.0).
        sources: Citation URLs supporting the claim.
    """

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)


class GroundedOutput(BaseModel):
    """Recommended base class for :meth:`DomainConfig.output_schema` results.

    Nodes that grade output (``FactCheckGate``, ``CriticNode``) operate on
    the ``claims`` attribute via duck typing, so subclasses may add extra
    fields freely as long as ``claims`` remains a list of ``Claim``-shaped
    items.
    """

    claims: list[Claim] = Field(default_factory=list)


class CriticVerdict(BaseModel):
    """The structured verdict returned by a :class:`JudgeLLM`.

    Attributes:
        verdict: ``"PASS"`` if the critic accepts the output, else ``"FAIL"``.
        rejected_claims: Claim texts to flag as previously-rejected on retry.
        reason: Free-form explanation. Never injected into prompts directly.
    """

    verdict: Literal["PASS", "FAIL"]
    rejected_claims: list[str] = Field(default_factory=list)
    reason: str = ""
