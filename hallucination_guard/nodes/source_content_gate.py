"""SourceContentGate — verify each claim is supported by its source content.

The third tier of source verification, sitting after ``SourceFetchGate``.

Reachability (``SourceFetchGate``) only proves an HTTP request succeeds;
it cannot tell whether the body actually backs the claim. A fabricated
URL whose host passes the allow-list, whose path happens to resolve to
an unrelated page, would still pass. ``SourceContentGate`` closes that
hole by fetching the source page and asking a judge whether the body
supports the claim.

Two pluggable strategies are injected:

* :class:`ContentFetcher` — returns page text (or ``None`` when the URL
  is unreachable / unparseable).
* :class:`SupportJudge` — decides whether a passage supports a claim.
  Typically backed by a separate LLM with a domain-tuned prompt.

The node short-circuits per claim: as soon as one source passes both
fetch and judge, the claim is accepted, so the worst case is one
``fetch`` + one ``supports`` call per source until the first PASS.

Failures reuse :attr:`FailReason.NO_SOURCE` so custom ``DomainConfig``
subclasses' ``retry_instruction`` need no changes (same rationale as
``SourceFetchGate``).
"""

from __future__ import annotations

import html
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from hallucination_guard.exceptions import GraphError
from hallucination_guard.state import FailReason, GraphState


@runtime_checkable
class ContentFetcher(Protocol):
    """Fetches the textual content of a source URL.

    Implementations are responsible for transport, encoding, extraction,
    and any caching. Return ``None`` whenever the URL cannot be turned
    into a usable passage (unreachable host, non-2xx, parse failure,
    paywall, robots disallow, …); the node treats that case identically
    to a server error.
    """

    def fetch(self, url: str) -> str | None:
        """Return the page text, or ``None`` when no passage can be obtained."""
        ...


@runtime_checkable
class SupportJudge(Protocol):
    """Decides whether a fetched passage supports a specific claim."""

    def supports(self, claim: str, passage: str) -> bool:
        """Return ``True`` iff ``passage`` plausibly supports ``claim``."""
        ...


class SourceContentGate:
    """Per-claim content-level support check."""

    def __init__(self, fetcher: ContentFetcher, judge: SupportJudge) -> None:
        self.fetcher = fetcher
        self.judge = judge

    def __call__(self, state: GraphState) -> GraphState:
        output = state.research_output
        if output is None:
            raise GraphError("SourceContentGate received empty research_output")

        claims = self._extract_claims(output)

        unsupported = [c for c in claims if not self._claim_supported(c)]
        if unsupported:
            additions = [
                f"{FailReason.NO_SOURCE.value}:{c.text}" for c in unsupported
            ]
            return state.with_update(
                gate_result="FAIL",
                fail_reason=FailReason.NO_SOURCE,
                fail_history=state.fail_history + additions,
            )

        return state.with_update(gate_result="PASS")

    def _claim_supported(self, claim: Any) -> bool:
        if not claim.sources:
            return False
        for url in claim.sources:
            passage = self.fetcher.fetch(url)
            if passage is None or not passage.strip():
                continue
            if self.judge.supports(claim.text, passage):
                return True
        return False

    @staticmethod
    def _extract_claims(output: Any) -> list[Any]:
        if not hasattr(output, "claims"):
            raise GraphError(
                "research_output must expose a `.claims` attribute; "
                f"got {type(output).__name__}"
            )
        return list(output.claims)


_DEFAULT_ACCEPT_STATUS: frozenset[int] = frozenset(range(200, 300))
_DEFAULT_MAX_BYTES = 256 * 1024
_DEFAULT_MAX_CHARS = 8 * 1024


class _TextExtractor(HTMLParser):
    """Collects visible text from an HTML stream, dropping script/style."""

    _SKIP_TAGS = frozenset({"script", "style", "head", "noscript"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join(part.strip() for part in self._chunks if part.strip())


class HTTPContentFetcher:
    """Stdlib-only :class:`ContentFetcher` using ``urllib.request``.

    Issues a GET, decodes the body as UTF-8 (with replacement on errors),
    strips HTML tags through :class:`html.parser.HTMLParser`, and
    returns at most ``max_chars`` characters of visible text. Reads are
    bounded by ``max_bytes`` to keep memory and judge-prompt size in
    check.

    No third-party dependency. Callers who need richer extraction
    (``trafilatura``, ``readability-lxml``, BeautifulSoup, …) or
    request caching can implement :class:`ContentFetcher` directly.
    """

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        accept_status: frozenset[int] = _DEFAULT_ACCEPT_STATUS,
        user_agent: str = "hallguard-source-check/1.0",
        max_bytes: int = _DEFAULT_MAX_BYTES,
        max_chars: int = _DEFAULT_MAX_CHARS,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.timeout = timeout
        self.accept_status = accept_status
        self.user_agent = user_agent
        self.max_bytes = max_bytes
        self.max_chars = max_chars

    def fetch(self, url: str) -> str | None:
        if not url:
            return None
        try:
            parsed = urlparse(url)
        except ValueError:
            return None
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return None

        req = urllib.request.Request(
            url, method="GET", headers={"User-Agent": self.user_agent}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status not in self.accept_status:
                    return None
                raw = resp.read(self.max_bytes + 1)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None

        body = raw[: self.max_bytes].decode("utf-8", errors="replace")
        text = self._extract_text(body)
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        return text or None

    @staticmethod
    def _extract_text(body: str) -> str:
        parser = _TextExtractor()
        try:
            parser.feed(body)
            parser.close()
        except Exception:
            return html.unescape(body).strip()
        return parser.text()
