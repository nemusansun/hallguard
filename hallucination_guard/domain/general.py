"""GeneralDomain — a permissive default :class:`DomainConfig` for generic Q&A.

Suitable as a baseline for examples and tests. Stricter domains (medical,
legal, …) should subclass :class:`DomainConfig` directly and tighten the
confidence threshold, source allow-list, and critic prompt.

Set ``locale="ja"`` at construction time to swap the English prompts for
their Japanese counterparts; default is ``"en"`` so existing callers see
no change in behavior.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import BaseModel

from hallucination_guard.domain.base import DomainConfig, Locale
from hallucination_guard.retry.directive import RetryDirective
from hallucination_guard.schemas import GroundedOutput


_GENERAL_SYSTEM_PROMPT_EN = """\
You are a careful research assistant. Answer the user's query using only
information you can cite. For every claim, attach a confidence score in
[0.0, 1.0] and at least one source URL. Prefer abstaining over guessing.
"""


_GENERAL_SYSTEM_PROMPT_JA = """\
あなたは慎重なリサーチアシスタントです。ユーザーの質問には、出典を
示せる情報のみを用いて回答してください。各主張には [0.0, 1.0] の
範囲の確信度と少なくとも 1 つのソース URL を必ず添えてください。
推測するくらいなら回答を控えてください。
"""


_GENERAL_CRITIC_PROMPT_EN = """\
You are a meticulous fact-checker. Read the candidate output and decide
whether it contains internal contradictions or claims that are not
plausibly supported by their citations.

Return verdict=PASS only if all claims are mutually consistent and each
claim is plausibly supported by at least one of its sources. Otherwise
return verdict=FAIL and list the offending claim texts in rejected_claims.
Do not invent claims that are not in the candidate output.
"""


_GENERAL_CRITIC_PROMPT_JA = """\
あなたは緻密なファクトチェッカーです。候補出力を読み、内部矛盾や、
引用元によって妥当に裏付けられていない主張がないか判定してください。

すべての主張が相互に整合し、各主張が少なくとも 1 つの出典で妥当に
裏付けられている場合にのみ verdict=PASS を返してください。それ以外
の場合は verdict=FAIL を返し、問題のある主張文を rejected_claims に
原文のまま列挙してください。候補出力に存在しない主張を捏造しないで
ください。
"""


class GeneralDomain(DomainConfig):
    """Permissive default domain.

    - Confidence threshold: ``0.7``
    - Sources: any ``https://`` URL with a non-empty host
    - Critic prompt: generic consistency / support check
    - Output schema: :class:`GroundedOutput`
    - Locale: ``"en"`` (default) or ``"ja"``
    """

    def __init__(self, *, locale: Locale = "en") -> None:
        self.locale: Locale = locale

    def retry_locale(self) -> Locale:
        return self.locale

    @property
    def confidence_threshold(self) -> float:
        return 0.7

    def is_valid_source(self, url: str) -> bool:
        if not url:
            return False
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        return parsed.scheme == "https" and bool(parsed.netloc)

    def critic_prompt(self) -> str:
        if self.locale == "ja":
            return _GENERAL_CRITIC_PROMPT_JA
        return _GENERAL_CRITIC_PROMPT_EN

    def output_schema(self) -> type[BaseModel]:
        return GroundedOutput

    def system_prompt(self) -> str:
        if self.locale == "ja":
            return _GENERAL_SYSTEM_PROMPT_JA
        return _GENERAL_SYSTEM_PROMPT_EN

    def format_retry_directive(
        self, base_prompt: str, directive: RetryDirective
    ) -> str:
        if self.locale == "ja":
            lines = [
                base_prompt,
                "",
                "--- 再試行指示 ---",
                directive.fix_instruction,
            ]
            if directive.forbidden_claims:
                lines.append("以下の却下済み主張は繰り返さないでください:")
                lines.extend(f"- {claim}" for claim in directive.forbidden_claims)
            return "\n".join(lines)

        lines = [
            base_prompt,
            "",
            "--- retry directive ---",
            directive.fix_instruction,
        ]
        if directive.forbidden_claims:
            lines.append("Do not repeat these previously-rejected claims:")
            lines.extend(f"- {claim}" for claim in directive.forbidden_claims)
        return "\n".join(lines)
