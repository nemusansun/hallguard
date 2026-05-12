"""DomainConfig — abstract Strategy interface for domain-specific behavior.

Framework code MUST NOT contain domain-specific knowledge (thresholds, source
validation rules, critic prompts, output schemas). Subclasses encapsulate that
knowledge and are injected at graph construction time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from hallucination_guard.retry.directive import RetryDirective


Locale = Literal["en", "ja"]
"""Supported prompt locales for built-in domains.

The framework itself stays locale-agnostic; ``Locale`` is exported so
subclasses (and tests) can type-check ``locale`` arguments uniformly when
they opt into multilingual prompts. Concrete domains are free to ignore it.
"""


class DomainConfig(ABC):
    """Abstract domain configuration consumed by the graph nodes.

    Implementations supply the confidence threshold, source validation,
    critic prompt, and structured output schema used by ``StructuredNode``,
    ``FactCheckGate``, and ``CriticNode``.
    """

    @property
    @abstractmethod
    def confidence_threshold(self) -> float:
        """Minimum confidence required to pass the FactCheckGate.

        Outputs whose confidence falls below this value are rejected and
        routed back through ``RetryNode``.
        """

    @abstractmethod
    def is_valid_source(self, url: str) -> bool:
        """Return ``True`` iff ``url`` is an acceptable citation for this domain.

        Used by ``FactCheckGate`` to enforce source grounding. Implementations
        decide what counts as a valid source (allow-listed domains, scheme
        checks, signed URLs, etc.).
        """

    @abstractmethod
    def critic_prompt(self) -> str:
        """Return the system prompt used by ``CriticNode``.

        The critic is a separate agent whose job is to detect contradictions
        or unsupported claims in the structured output.
        """

    @abstractmethod
    def output_schema(self) -> type[BaseModel]:
        """Return the Pydantic model that ``StructuredNode`` will enforce.

        Returning a model class (not an instance) lets the structured-output
        layer bind the schema directly to the LLM call.
        """

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the base system prompt used by ``StructuredNode``.

        The retry directive (if any) is appended by ``format_retry_directive``;
        this method returns only the domain-level instructions that should
        be active on every attempt.
        """

    @abstractmethod
    def format_retry_directive(
        self, base_prompt: str, directive: RetryDirective
    ) -> str:
        """Compose the system prompt for a retry attempt.

        Receives the base system prompt (as returned by ``system_prompt``)
        and a :class:`RetryDirective` and returns the full system prompt
        the retry should run with. Letting the domain own this assembly
        keeps wording — separators, forbidden-claims phrasing, language —
        out of ``StructuredNode`` so the framework core stays domain-agnostic.

        Implementations MUST NOT splice raw ``fail_history`` strings into
        the returned prompt; only ``directive.fix_instruction`` and
        ``directive.forbidden_claims`` are safe to emit.
        """

    def retry_locale(self) -> Locale:
        """Return the locale used to render the retry ``fix_instruction``.

        Defaults to ``"en"`` so subclasses that do not opt into multilingual
        prompts get English retry hints — matching the default of
        :meth:`RetryHintBuilder.build`. Domains that swap their other
        prompts by locale should override this so the hint builder stays
        aligned with the rest of the prompt surface.
        """
        return "en"
