"""SourceFetchGate — verify source URLs are reachable over HTTP.

A second-stage source check that complements ``FactCheckGate``.

``FactCheckGate`` only inspects the **string shape** of a source URL
(scheme, host, optional allow-list). It cannot tell whether the URL
actually resolves: a syntactically valid ``https://pubmed.ncbi.nlm.nih.gov/...``
that points at a fabricated article id still passes. ``SourceFetchGate``
closes that hole by performing a real HTTP request per source URL and
treating unreachable URLs as a ``NO_SOURCE`` failure, routing back
through ``RetryNode`` just like the string-shape check.

Wiring is opt-in: pass a ``SourceFetcher`` to :class:`Graph` to enable
the node. The default graph performs no network I/O so existing callers
are unaffected.

Failure rules mirror ``FactCheckGate``:

- Empty sources list → FAIL
- All sources unreachable → FAIL
- At least one source reachable → PASS

One ``fail_history`` entry is appended per failing claim, prefixed with
``FailReason.NO_SOURCE.value`` (consistent with ``FactCheckGate``) so
``get_rejected_claims`` and downstream logic see a single source-failure
category rather than two parallel ones.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from hallucination_guard.exceptions import GraphError
from hallucination_guard.state import FailReason, GraphState


@runtime_checkable
class SourceFetcher(Protocol):
    """Pluggable strategy for deciding whether a source URL is reachable.

    Implementations are free to use any transport (``urllib``, ``httpx``,
    a cached HEAD service, a signed-URL validator, …) and any caching
    policy. The node only requires a synchronous boolean answer.
    """

    def check(self, url: str) -> bool:
        """Return ``True`` iff ``url`` should be considered reachable."""
        ...


class SourceFetchGate:
    """Network-level source verification node."""

    def __init__(self, fetcher: SourceFetcher) -> None:
        self.fetcher = fetcher

    def __call__(self, state: GraphState) -> GraphState:
        output = state.research_output
        if output is None:
            raise GraphError("SourceFetchGate received empty research_output")

        claims = self._extract_claims(output)

        unreachable = [
            c
            for c in claims
            if not c.sources or not any(self.fetcher.check(s) for s in c.sources)
        ]
        if unreachable:
            additions = [
                f"{FailReason.NO_SOURCE.value}:{c.text}" for c in unreachable
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


_DEFAULT_ACCEPT_STATUS: frozenset[int] = frozenset(range(200, 400))


class HTTPHeadFetcher:
    """Stdlib-only :class:`SourceFetcher` using ``urllib.request``.

    Performs a HEAD request first and falls back to a small GET when the
    server returns 403 or 405 (common when HEAD is blocked). Any
    network-layer error, timeout, malformed URL, or status outside
    ``accept_status`` is treated as unreachable.

    No third-party dependency is pulled in. Callers who want ``httpx``,
    ``requests``, or custom caching can implement :class:`SourceFetcher`
    directly.
    """

    def __init__(
        self,
        *,
        timeout: float = 5.0,
        accept_status: frozenset[int] = _DEFAULT_ACCEPT_STATUS,
        user_agent: str = "hallguard-source-check/1.0",
        get_fallback_status: frozenset[int] = frozenset({403, 405}),
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self.accept_status = accept_status
        self.user_agent = user_agent
        self.get_fallback_status = get_fallback_status

    def check(self, url: str) -> bool:
        if not url:
            return False
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False

        status = self._request_status(url, method="HEAD")
        if status is not None and status in self.accept_status:
            return True
        if status is not None and status in self.get_fallback_status:
            status = self._request_status(url, method="GET")
            return status is not None and status in self.accept_status
        return False

    def _request_status(self, url: str, *, method: str) -> int | None:
        req = urllib.request.Request(
            url, method=method, headers={"User-Agent": self.user_agent}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                return int(status) if status is not None else None
        except urllib.error.HTTPError as exc:
            return int(exc.code)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None
