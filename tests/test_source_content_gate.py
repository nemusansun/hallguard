"""Tests for :mod:`hallucination_guard.nodes.source_content_gate`.

Node-level tests use recording stub fetcher / judge so no network is
touched. The ``HTTPContentFetcher`` tests stand up a local
``http.server`` so each status code and body shape can be exercised
deterministically.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator

import pytest

from hallucination_guard.exceptions import GraphError
from hallucination_guard.nodes.source_content_gate import (
    ContentFetcher,
    HTTPContentFetcher,
    SourceContentGate,
    SupportJudge,
)
from hallucination_guard.schemas import Claim, GroundedOutput
from hallucination_guard.state import FailReason, GraphState


# --- Node-level tests --------------------------------------------------------


class _RecordingFetcher:
    """``ContentFetcher`` whose passages are driven by an explicit dict."""

    def __init__(self, passages: dict[str, str | None]) -> None:
        self._passages = passages
        self.calls: list[str] = []

    def fetch(self, url: str) -> str | None:
        self.calls.append(url)
        return self._passages.get(url)


class _RecordingJudge:
    """``SupportJudge`` driven by a fixed (claim, passage) -> bool dict."""

    def __init__(self, verdicts: dict[tuple[str, str], bool]) -> None:
        self._verdicts = verdicts
        self.calls: list[tuple[str, str]] = []

    def supports(self, claim: str, passage: str) -> bool:
        self.calls.append((claim, passage))
        return self._verdicts.get((claim, passage), False)


def _state_with(*claims: Claim) -> GraphState:
    return GraphState(
        user_query="q",
        research_output=GroundedOutput(claims=list(claims)),
    )


def test_recording_stubs_satisfy_protocols() -> None:
    assert isinstance(_RecordingFetcher({}), ContentFetcher)
    assert isinstance(_RecordingJudge({}), SupportJudge)


def test_passes_when_passage_supports_claim() -> None:
    fetcher = _RecordingFetcher({"https://a.example/1": "water boils at 100C"})
    judge = _RecordingJudge({("water boils", "water boils at 100C"): True})
    gate = SourceContentGate(fetcher, judge)
    state = _state_with(
        Claim(text="water boils", confidence=0.9, sources=["https://a.example/1"]),
    )
    result = gate(state)
    assert result.gate_result == "PASS"
    assert result.fail_reason is None
    assert result.fail_history == []


def test_fails_when_passage_does_not_support_claim() -> None:
    fetcher = _RecordingFetcher({"https://a.example/1": "an unrelated paragraph"})
    judge = _RecordingJudge({})  # default verdict is False
    gate = SourceContentGate(fetcher, judge)
    state = _state_with(
        Claim(text="X", confidence=0.9, sources=["https://a.example/1"]),
    )
    result = gate(state)
    assert result.gate_result == "FAIL"
    assert result.fail_reason == FailReason.NO_SOURCE
    assert result.fail_history == ["no_source:X"]


def test_fails_when_claim_has_no_sources() -> None:
    fetcher = _RecordingFetcher({})
    judge = _RecordingJudge({})
    gate = SourceContentGate(fetcher, judge)
    state = _state_with(Claim(text="empty", confidence=0.9, sources=[]))
    result = gate(state)
    assert result.gate_result == "FAIL"
    assert result.fail_history == ["no_source:empty"]
    assert fetcher.calls == []
    assert judge.calls == []


def test_fails_when_every_source_fetch_returns_none() -> None:
    fetcher = _RecordingFetcher({"https://a/1": None, "https://b/2": None})
    judge = _RecordingJudge({})
    gate = SourceContentGate(fetcher, judge)
    state = _state_with(
        Claim(text="X", confidence=0.9, sources=["https://a/1", "https://b/2"]),
    )
    result = gate(state)
    assert result.gate_result == "FAIL"
    # Judge must not be asked when the fetcher returns no passage.
    assert judge.calls == []


def test_fails_when_fetcher_returns_blank_strings() -> None:
    fetcher = _RecordingFetcher({"https://a/1": "   "})
    judge = _RecordingJudge({})
    gate = SourceContentGate(fetcher, judge)
    state = _state_with(
        Claim(text="X", confidence=0.9, sources=["https://a/1"]),
    )
    result = gate(state)
    assert result.gate_result == "FAIL"
    assert judge.calls == []  # blank passage skipped


def test_short_circuits_on_first_supporting_source() -> None:
    fetcher = _RecordingFetcher(
        {
            "https://a/1": "first body",
            "https://b/2": "second body",
            "https://c/3": "third body",
        }
    )
    judge = _RecordingJudge({("X", "first body"): True})
    gate = SourceContentGate(fetcher, judge)
    state = _state_with(
        Claim(
            text="X",
            confidence=0.9,
            sources=["https://a/1", "https://b/2", "https://c/3"],
        ),
    )
    result = gate(state)
    assert result.gate_result == "PASS"
    # Once the first source supports the claim, remaining sources are skipped.
    assert fetcher.calls == ["https://a/1"]
    assert judge.calls == [("X", "first body")]


def test_passes_when_a_later_source_supports_claim() -> None:
    fetcher = _RecordingFetcher(
        {
            "https://a/1": "irrelevant body",
            "https://b/2": "supporting body",
        }
    )
    judge = _RecordingJudge({("X", "supporting body"): True})
    gate = SourceContentGate(fetcher, judge)
    state = _state_with(
        Claim(
            text="X",
            confidence=0.9,
            sources=["https://a/1", "https://b/2"],
        ),
    )
    result = gate(state)
    assert result.gate_result == "PASS"
    assert fetcher.calls == ["https://a/1", "https://b/2"]


def test_appends_one_history_entry_per_failing_claim() -> None:
    fetcher = _RecordingFetcher(
        {"https://a/1": "body A", "https://b/2": "body B", "https://c/3": "body C"}
    )
    judge = _RecordingJudge({("C", "body C"): True})
    gate = SourceContentGate(fetcher, judge)
    state = _state_with(
        Claim(text="A", confidence=0.9, sources=["https://a/1"]),
        Claim(text="B", confidence=0.9, sources=["https://b/2"]),
        Claim(text="C", confidence=0.9, sources=["https://c/3"]),
    )
    result = gate(state)
    assert result.fail_history == ["no_source:A", "no_source:B"]


def test_preserves_existing_fail_history() -> None:
    fetcher = _RecordingFetcher({"https://a/1": "body"})
    judge = _RecordingJudge({})
    gate = SourceContentGate(fetcher, judge)
    state = GraphState(
        user_query="q",
        research_output=GroundedOutput(
            claims=[Claim(text="x", confidence=0.9, sources=["https://a/1"])]
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
    judge = _RecordingJudge({})
    gate = SourceContentGate(fetcher, judge)
    state = GraphState(user_query="q", research_output=None)
    with pytest.raises(GraphError):
        gate(state)


def test_does_not_mutate_input_state() -> None:
    fetcher = _RecordingFetcher({"https://a/1": "body"})
    judge = _RecordingJudge({})
    gate = SourceContentGate(fetcher, judge)
    state = _state_with(
        Claim(text="x", confidence=0.9, sources=["https://a/1"]),
    )
    original_history = list(state.fail_history)
    gate(state)
    assert state.fail_history == original_history
    assert state.gate_result is None


# --- HTTPContentFetcher tests ------------------------------------------------


_HTML_BODY = (
    b"<html><head><title>t</title>"
    b"<style>body{color:red}</style>"
    b"<script>var x=1;</script>"
    b"</head><body>"
    b"<h1>Heading</h1>"
    b"<p>First paragraph &amp; second.</p>"
    b"<script>alert('x')</script>"
    b"<p>More text.</p>"
    b"</body></html>"
)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server API
        if self.path == "/page":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_HTML_BODY)))
            self.end_headers()
            self.wfile.write(_HTML_BODY)
            return
        if self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return
        if self.path == "/empty":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>  </body></html>")
            return
        if self.path == "/huge":
            payload = b"<p>" + (b"a" * 200_000) + b"</p>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


@pytest.fixture(scope="module")
def stub_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2.0)


def test_http_content_fetcher_returns_visible_text(stub_server: str) -> None:
    fetcher = HTTPContentFetcher(timeout=2.0)
    body = fetcher.fetch(f"{stub_server}/page")
    assert body is not None
    assert "Heading" in body
    assert "First paragraph & second." in body
    assert "More text." in body
    # Script / style content must be stripped.
    assert "alert" not in body
    assert "color:red" not in body


def test_http_content_fetcher_returns_none_on_404(stub_server: str) -> None:
    fetcher = HTTPContentFetcher(timeout=2.0)
    assert fetcher.fetch(f"{stub_server}/missing") is None


def test_http_content_fetcher_returns_none_on_empty_body(stub_server: str) -> None:
    fetcher = HTTPContentFetcher(timeout=2.0)
    assert fetcher.fetch(f"{stub_server}/empty") is None


def test_http_content_fetcher_truncates_to_max_chars(stub_server: str) -> None:
    fetcher = HTTPContentFetcher(timeout=2.0, max_chars=100, max_bytes=5_000)
    body = fetcher.fetch(f"{stub_server}/huge")
    assert body is not None
    assert len(body) <= 100


def test_http_content_fetcher_rejects_empty_url() -> None:
    fetcher = HTTPContentFetcher(timeout=1.0)
    assert fetcher.fetch("") is None


def test_http_content_fetcher_rejects_non_http_scheme() -> None:
    fetcher = HTTPContentFetcher(timeout=1.0)
    assert fetcher.fetch("ftp://example.com/x") is None
    assert fetcher.fetch("file:///etc/passwd") is None


def test_http_content_fetcher_rejects_unresolvable_host() -> None:
    fetcher = HTTPContentFetcher(timeout=1.0)
    assert fetcher.fetch("https://nope.invalid/x") is None


def test_http_content_fetcher_rejects_invalid_constructor_args() -> None:
    with pytest.raises(ValueError):
        HTTPContentFetcher(timeout=0)
    with pytest.raises(ValueError):
        HTTPContentFetcher(max_bytes=0)
    with pytest.raises(ValueError):
        HTTPContentFetcher(max_chars=0)
