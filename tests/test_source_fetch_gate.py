"""Tests for :mod:`hallucination_guard.nodes.source_fetch_gate`.

The node-level tests use a recording stub ``SourceFetcher`` so no network
is touched. The ``HTTPHeadFetcher`` tests stand up a local
``http.server`` so each status code can be exercised deterministically.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator

import pytest

from hallucination_guard.exceptions import GraphError
from hallucination_guard.nodes.source_fetch_gate import (
    HTTPHeadFetcher,
    SourceFetcher,
    SourceFetchGate,
)
from hallucination_guard.schemas import Claim, GroundedOutput
from hallucination_guard.state import FailReason, GraphState


# --- Node-level tests --------------------------------------------------------


class _RecordingFetcher:
    """``SourceFetcher`` whose verdicts are driven by an explicit dict."""

    def __init__(self, verdicts: dict[str, bool]) -> None:
        self._verdicts = verdicts
        self.calls: list[str] = []

    def check(self, url: str) -> bool:
        self.calls.append(url)
        return self._verdicts.get(url, False)


def _state_with(*claims: Claim) -> GraphState:
    return GraphState(
        user_query="q",
        research_output=GroundedOutput(claims=list(claims)),
    )


def test_recording_fetcher_satisfies_protocol() -> None:
    assert isinstance(_RecordingFetcher({}), SourceFetcher)


def test_passes_when_every_claim_has_a_reachable_source() -> None:
    fetcher = _RecordingFetcher(
        {"https://a.example/1": True, "https://b.example/2": True}
    )
    gate = SourceFetchGate(fetcher)
    state = _state_with(
        Claim(text="A", confidence=0.9, sources=["https://a.example/1"]),
        Claim(text="B", confidence=0.9, sources=["https://b.example/2"]),
    )
    result = gate(state)
    assert result.gate_result == "PASS"
    assert result.fail_reason is None
    assert result.fail_history == []


def test_fails_when_every_source_is_unreachable() -> None:
    fetcher = _RecordingFetcher({"https://broken.example/1": False})
    gate = SourceFetchGate(fetcher)
    state = _state_with(
        Claim(text="bad", confidence=0.9, sources=["https://broken.example/1"]),
    )
    result = gate(state)
    assert result.gate_result == "FAIL"
    assert result.fail_reason == FailReason.NO_SOURCE
    assert result.fail_history == ["no_source:bad"]


def test_fails_when_claim_has_no_sources() -> None:
    fetcher = _RecordingFetcher({})
    gate = SourceFetchGate(fetcher)
    state = _state_with(Claim(text="empty", confidence=0.9, sources=[]))
    result = gate(state)
    assert result.gate_result == "FAIL"
    assert result.fail_reason == FailReason.NO_SOURCE
    assert result.fail_history == ["no_source:empty"]
    assert fetcher.calls == []


def test_passes_when_at_least_one_source_is_reachable() -> None:
    fetcher = _RecordingFetcher(
        {"https://dead.example/1": False, "https://live.example/2": True}
    )
    gate = SourceFetchGate(fetcher)
    state = _state_with(
        Claim(
            text="mixed",
            confidence=0.9,
            sources=["https://dead.example/1", "https://live.example/2"],
        ),
    )
    result = gate(state)
    assert result.gate_result == "PASS"


def test_appends_one_history_entry_per_failing_claim() -> None:
    fetcher = _RecordingFetcher({"https://ok.example": True})
    gate = SourceFetchGate(fetcher)
    state = _state_with(
        Claim(text="A", confidence=0.9, sources=["https://nope.example/1"]),
        Claim(text="B", confidence=0.9, sources=["https://nope.example/2"]),
        Claim(text="C", confidence=0.9, sources=["https://ok.example"]),
    )
    result = gate(state)
    assert result.fail_history == ["no_source:A", "no_source:B"]


def test_preserves_existing_fail_history() -> None:
    fetcher = _RecordingFetcher({})
    gate = SourceFetchGate(fetcher)
    state = GraphState(
        user_query="q",
        research_output=GroundedOutput(
            claims=[Claim(text="x", confidence=0.9, sources=["https://no.example"])]
        ),
        fail_history=["critic_rejected:earlier"],
    )
    result = gate(state)
    assert result.fail_history == [
        "critic_rejected:earlier",
        "no_source:x",
    ]


def test_raises_when_research_output_is_missing() -> None:
    fetcher = _RecordingFetcher({})
    gate = SourceFetchGate(fetcher)
    state = GraphState(user_query="q", research_output=None)
    with pytest.raises(GraphError):
        gate(state)


def test_does_not_mutate_input_state() -> None:
    fetcher = _RecordingFetcher({})
    gate = SourceFetchGate(fetcher)
    state = _state_with(
        Claim(text="x", confidence=0.9, sources=["https://no.example"]),
    )
    original_history = list(state.fail_history)
    gate(state)
    assert state.fail_history == original_history
    assert state.gate_result is None


# --- HTTPHeadFetcher tests ---------------------------------------------------


_STATUS_RESPONSES: dict[str, int] = {
    "/ok": 200,
    "/redirect": 302,
    "/missing": 404,
    "/server-error": 500,
    "/head-blocked": 405,
    "/forbidden-head": 403,
}


class _StubHandler(BaseHTTPRequestHandler):
    """Maps paths to fixed status codes. Path semantics shared across methods."""

    def do_HEAD(self) -> None:  # noqa: N802 — http.server API
        status = _STATUS_RESPONSES.get(self.path, 404)
        if self.path in {"/head-blocked", "/forbidden-head"}:
            self.send_response(status)
            self.end_headers()
            return
        self.send_response(status)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 — http.server API
        if self.path in {"/head-blocked", "/forbidden-head"}:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        status = _STATUS_RESPONSES.get(self.path, 404)
        self.send_response(status)
        self.end_headers()
        self.wfile.write(b"x")

    def log_message(self, *_args: object) -> None:  # silence test output
        return


@pytest.fixture(scope="module")
def stub_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2.0)


def test_http_head_fetcher_accepts_200(stub_server: str) -> None:
    fetcher = HTTPHeadFetcher(timeout=2.0)
    assert fetcher.check(f"{stub_server}/ok") is True


def test_http_head_fetcher_accepts_redirect(stub_server: str) -> None:
    fetcher = HTTPHeadFetcher(timeout=2.0)
    assert fetcher.check(f"{stub_server}/redirect") is True


def test_http_head_fetcher_rejects_404(stub_server: str) -> None:
    fetcher = HTTPHeadFetcher(timeout=2.0)
    assert fetcher.check(f"{stub_server}/missing") is False


def test_http_head_fetcher_rejects_5xx(stub_server: str) -> None:
    fetcher = HTTPHeadFetcher(timeout=2.0)
    assert fetcher.check(f"{stub_server}/server-error") is False


def test_http_head_fetcher_falls_back_to_get_on_405(stub_server: str) -> None:
    fetcher = HTTPHeadFetcher(timeout=2.0)
    assert fetcher.check(f"{stub_server}/head-blocked") is True


def test_http_head_fetcher_falls_back_to_get_on_403(stub_server: str) -> None:
    fetcher = HTTPHeadFetcher(timeout=2.0)
    assert fetcher.check(f"{stub_server}/forbidden-head") is True


def test_http_head_fetcher_rejects_empty_url() -> None:
    fetcher = HTTPHeadFetcher(timeout=1.0)
    assert fetcher.check("") is False


def test_http_head_fetcher_rejects_non_http_scheme() -> None:
    fetcher = HTTPHeadFetcher(timeout=1.0)
    assert fetcher.check("ftp://example.com/x") is False
    assert fetcher.check("file:///etc/passwd") is False


def test_http_head_fetcher_rejects_unresolvable_host() -> None:
    fetcher = HTTPHeadFetcher(timeout=1.0)
    # Reserved TLD `.invalid` is guaranteed not to resolve (RFC 6761).
    assert fetcher.check("https://nope.invalid/x") is False


def test_http_head_fetcher_rejects_zero_timeout() -> None:
    with pytest.raises(ValueError):
        HTTPHeadFetcher(timeout=0)


def test_http_head_fetcher_accept_status_is_configurable(
    stub_server: str,
) -> None:
    strict = HTTPHeadFetcher(timeout=2.0, accept_status=frozenset({200}))
    assert strict.check(f"{stub_server}/redirect") is False
    assert strict.check(f"{stub_server}/ok") is True
