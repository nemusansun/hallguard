"""MedicalDomain — a strict :class:`DomainConfig` for medical questions.

A demonstration of how a domain can tighten the framework's safety knobs:

- Confidence threshold raised to ``0.95``
- Citations restricted to a small allow-list of well-known medical sources
- Critic prompt that explicitly forbids unsupported clinical guidance

The allow-list is intentionally tiny; production deployments are expected
to override it with their own institutional sources.

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


_ALLOWED_HOSTS = frozenset(
    {
        "pubmed.ncbi.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "www.who.int",
        "who.int",
        "www.cdc.gov",
        "cdc.gov",
        "www.cochranelibrary.com",
        "cochranelibrary.com",
        "www.nejm.org",
        "nejm.org",
    }
)


_MEDICAL_SYSTEM_PROMPT_EN = """\
You are a clinical research assistant. Answer the user's medical question
using only information you can cite to a peer-reviewed or institutional
medical source. For every claim, attach a confidence score in [0.0, 1.0]
and at least one source URL. If the evidence is weak or absent, abstain
rather than guess. Never provide definitive treatment, dose, or diagnosis
advice without an explicit citation.
"""


_MEDICAL_SYSTEM_PROMPT_JA = """\
あなたは臨床研究アシスタントです。ユーザーの医療に関する質問には、
査読済みまたは公的医療機関の出典を引用できる情報のみを用いて回答
してください。各主張には [0.0, 1.0] の範囲の確信度と少なくとも 1 つの
ソース URL を必ず添えてください。エビデンスが乏しい、または存在
しない場合は推測せず回答を控えてください。明示的な引用なしに、
確定的な治療・用量・診断の助言を与えないでください。
"""


_MEDICAL_CRITIC_PROMPT_EN = """\
You are reviewing clinical or biomedical claims for safety. Reject the
output if any claim states a treatment, dose, or diagnosis without an
explicit citation to a reputable medical source, or if any two claims
contradict each other. Reject any claim that gives definitive clinical
guidance without acknowledging uncertainty.

Return verdict=PASS only when every claim is internally consistent and
plausibly supported by its cited sources. Otherwise return verdict=FAIL
and list the offending claim texts in rejected_claims verbatim.
"""


_MEDICAL_CRITIC_PROMPT_JA = """\
あなたは臨床・生物医学的主張の安全性レビュアーです。信頼できる医療
出典への明示的な引用なしに治療・用量・診断を述べている主張、ある
いは 2 つの主張が互いに矛盾している場合、その出力を却下してください。
不確実性に言及せずに確定的な臨床助言を与える主張も却下してください。

すべての主張が内部的に整合し、引用された出典で妥当に裏付けられて
いる場合にのみ verdict=PASS を返してください。それ以外の場合は
verdict=FAIL を返し、問題のある主張文を rejected_claims に原文のまま
列挙してください。
"""


class MedicalDomain(DomainConfig):
    """Strict medical-question domain.

    - Confidence threshold: ``0.95``
    - Sources: ``https://`` URLs whose host is in a small allow-list
      (PubMed, WHO, CDC, Cochrane, NEJM)
    - Critic prompt: rejects unsupported clinical guidance and contradictions
    - Output schema: :class:`GroundedOutput`
    - Locale: ``"en"`` (default) or ``"ja"``
    """

    def __init__(self, *, locale: Locale = "en") -> None:
        self.locale: Locale = locale

    def retry_locale(self) -> Locale:
        return self.locale

    @property
    def confidence_threshold(self) -> float:
        return 0.95

    def is_valid_source(self, url: str) -> bool:
        if not url:
            return False
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        if parsed.scheme != "https" or not parsed.netloc:
            return False
        return parsed.netloc.lower() in _ALLOWED_HOSTS

    def critic_prompt(self) -> str:
        if self.locale == "ja":
            return _MEDICAL_CRITIC_PROMPT_JA
        return _MEDICAL_CRITIC_PROMPT_EN

    def output_schema(self) -> type[BaseModel]:
        return GroundedOutput

    def system_prompt(self) -> str:
        if self.locale == "ja":
            return _MEDICAL_SYSTEM_PROMPT_JA
        return _MEDICAL_SYSTEM_PROMPT_EN

    def format_retry_directive(
        self, base_prompt: str, directive: RetryDirective
    ) -> str:
        if self.locale == "ja":
            lines = [
                base_prompt,
                "",
                "--- 再試行指示 (medical) ---",
                directive.fix_instruction,
                "各主張について、査読済みまたは公的医療機関の出典 "
                "(PubMed, WHO, CDC, Cochrane, NEJM) を引用してください。",
            ]
            if directive.forbidden_claims:
                lines.append(
                    "以下の却下済み臨床主張は再度述べないでください:"
                )
                lines.extend(f"- {claim}" for claim in directive.forbidden_claims)
            return "\n".join(lines)

        lines = [
            base_prompt,
            "",
            "--- retry directive (medical) ---",
            directive.fix_instruction,
            "Cite a peer-reviewed or institutional medical source "
            "(PubMed, WHO, CDC, Cochrane, NEJM) for every claim.",
        ]
        if directive.forbidden_claims:
            lines.append(
                "Do not restate any of the following previously-rejected "
                "clinical claims:"
            )
            lines.extend(f"- {claim}" for claim in directive.forbidden_claims)
        return "\n".join(lines)
