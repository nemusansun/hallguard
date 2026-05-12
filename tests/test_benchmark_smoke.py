"""Smoke test for :mod:`benchmarks.hallucination_rate`.

Keeps the benchmark importable and runnable so a refactor of the framework
core can't silently break the headline demo number.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from benchmarks import hallucination_rate


def test_benchmark_main_runs_with_simulated_llm(
    capsys: object,
) -> None:
    # Sanity-check the simulator-backed default path end-to-end.
    buf = io.StringIO()
    stdout = sys.stdout
    sys.stdout = buf
    try:
        exit_code = hallucination_rate.main(["--max-retries", "1"])
    finally:
        sys.stdout = stdout

    output = buf.getvalue()
    assert exit_code == 0
    assert "Hallucination-Guard benchmark" in output
    assert "queries" in output
    assert "successes" in output


def test_benchmark_verbose_emits_per_query_lines() -> None:
    buf = io.StringIO()
    stdout = sys.stdout
    sys.stdout = buf
    try:
        hallucination_rate.main(["--verbose"])
    finally:
        sys.stdout = stdout

    output = buf.getvalue()
    assert "Per-query results:" in output
    # At least one PASS and one FAIL should appear given the seeded dataset.
    assert "[PASS]" in output
    assert "[FAIL]" in output


def test_load_dataset_reads_bundled_default() -> None:
    dataset = hallucination_rate._load_dataset(hallucination_rate._DEFAULT_DATASET)
    # The default ships at least one EASY case so a no-arg run isn't trivially empty.
    assert len(dataset.cases) > 0
    queries = {c.query for c in dataset.cases}
    assert any("[EASY]" in q for q in queries)
    # The bundled dataset declares its intended domain so a wrong --domain
    # selection is detectable.
    assert dataset.suggested_domain == "general"


def test_load_dataset_accepts_custom_path(tmp_path: Path) -> None:
    custom = tmp_path / "mini.json"
    custom.write_text(
        json.dumps(
            {
                "cases": [
                    {"query": "[EASY] Two plus two?", "tag": "[EASY]"},
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = hallucination_rate._load_dataset(custom)
    assert dataset.cases == (
        hallucination_rate._Case(query="[EASY] Two plus two?", tag="[EASY]"),
    )
    # Datasets without suggested_domain leave the field as None.
    assert dataset.suggested_domain is None


def test_load_dataset_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        hallucination_rate._load_dataset(missing)


def test_load_dataset_raises_on_missing_cases_field(tmp_path: Path) -> None:
    broken = tmp_path / "bad.json"
    broken.write_text(json.dumps({"version": "1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="'cases'"):
        hallucination_rate._load_dataset(broken)


def test_load_dataset_raises_on_empty_query(tmp_path: Path) -> None:
    broken = tmp_path / "bad.json"
    broken.write_text(
        json.dumps({"cases": [{"query": "", "tag": "[EASY]"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'query'"):
        hallucination_rate._load_dataset(broken)


def test_benchmark_main_accepts_dataset_flag(tmp_path: Path) -> None:
    dataset = tmp_path / "single.json"
    dataset.write_text(
        json.dumps(
            {
                "cases": [
                    {"query": "[EASY] What is 1+1?", "tag": "[EASY]"},
                ]
            }
        ),
        encoding="utf-8",
    )
    buf = io.StringIO()
    stdout = sys.stdout
    sys.stdout = buf
    try:
        exit_code = hallucination_rate.main(
            ["--dataset", str(dataset), "--max-retries", "1"]
        )
    finally:
        sys.stdout = stdout

    output = buf.getvalue()
    assert exit_code == 0
    assert "queries    : 1" in output


def test_medical_dataset_ships_with_default_install() -> None:
    medical = (
        hallucination_rate._DEFAULT_DATASET.parent / "medical_qa.json"
    )
    assert medical.exists(), "medical_qa.json should ship in benchmarks/datasets/"
    dataset = hallucination_rate._load_dataset(medical)
    assert len(dataset.cases) > 0
    assert any("[EASY]" in c.query for c in dataset.cases)
    assert dataset.suggested_domain == "medical"


def test_benchmark_medical_domain_matches_general_aggregates() -> None:
    medical = (
        hallucination_rate._DEFAULT_DATASET.parent / "medical_qa.json"
    )
    buf = io.StringIO()
    stdout = sys.stdout
    sys.stdout = buf
    try:
        exit_code = hallucination_rate.main(
            ["--domain", "medical", "--dataset", str(medical)]
        )
    finally:
        sys.stdout = stdout

    output = buf.getvalue()
    assert exit_code == 0
    # Same headline number as the general benchmark — the deterministic
    # simulator pairs each dataset with a domain-valid good_source so the
    # tag-driven routing reproduces identical aggregates.
    assert "successes        :   3 / 7" in output
    assert "avg retry_count  : 1.86" in output


def test_benchmark_rejects_unknown_domain() -> None:
    buf = io.StringIO()
    stdout = sys.stdout
    sys.stdout = buf
    try:
        with pytest.raises(SystemExit):
            hallucination_rate.main(["--domain", "legal"])
    finally:
        sys.stdout = stdout


def test_main_warns_when_domain_mismatches_suggested_domain(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "tagged.json"
    dataset.write_text(
        json.dumps(
            {
                "suggested_domain": "medical",
                "cases": [
                    {"query": "[EASY] anything?", "tag": "[EASY]"},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_buf, err_buf
    try:
        exit_code = hallucination_rate.main(
            ["--dataset", str(dataset), "--domain", "general", "--max-retries", "1"]
        )
    finally:
        sys.stdout, sys.stderr = stdout, stderr

    assert exit_code == 0
    err = err_buf.getvalue()
    assert "WARNING" in err
    assert "suggested_domain='medical'" in err
    assert "--domain is 'general'" in err
    # Aggregate output still goes to stdout, untouched.
    assert "Hallucination-Guard benchmark" in out_buf.getvalue()


def test_main_does_not_warn_when_domain_matches_suggested_domain(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "matched.json"
    dataset.write_text(
        json.dumps(
            {
                "suggested_domain": "general",
                "cases": [
                    {"query": "[EASY] anything?", "tag": "[EASY]"},
                ],
            }
        ),
        encoding="utf-8",
    )

    err_buf = io.StringIO()
    stderr = sys.stderr
    sys.stderr = err_buf
    try:
        with __import__("contextlib").redirect_stdout(io.StringIO()):
            exit_code = hallucination_rate.main(
                ["--dataset", str(dataset), "--domain", "general", "--max-retries", "1"]
            )
    finally:
        sys.stderr = stderr

    assert exit_code == 0
    assert "WARNING" not in err_buf.getvalue()


def test_main_does_not_warn_when_suggested_domain_absent(tmp_path: Path) -> None:
    dataset = tmp_path / "no_hint.json"
    dataset.write_text(
        json.dumps(
            {
                "cases": [
                    {"query": "[EASY] anything?", "tag": "[EASY]"},
                ],
            }
        ),
        encoding="utf-8",
    )

    err_buf = io.StringIO()
    stderr = sys.stderr
    sys.stderr = err_buf
    try:
        with __import__("contextlib").redirect_stdout(io.StringIO()):
            exit_code = hallucination_rate.main(
                ["--dataset", str(dataset), "--domain", "medical", "--max-retries", "1"]
            )
    finally:
        sys.stderr = stderr

    assert exit_code == 0
    assert "WARNING" not in err_buf.getvalue()


def test_load_dataset_rejects_non_string_suggested_domain(tmp_path: Path) -> None:
    broken = tmp_path / "bad.json"
    broken.write_text(
        json.dumps(
            {
                "suggested_domain": 42,
                "cases": [{"query": "[EASY] q?", "tag": "[EASY]"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="suggested_domain"):
        hallucination_rate._load_dataset(broken)


def test_strict_domain_exits_nonzero_on_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "tagged.json"
    dataset.write_text(
        json.dumps(
            {
                "suggested_domain": "medical",
                "cases": [
                    {"query": "[EASY] anything?", "tag": "[EASY]"},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_buf, err_buf
    try:
        exit_code = hallucination_rate.main(
            [
                "--dataset",
                str(dataset),
                "--domain",
                "general",
                "--strict-domain",
                "--max-retries",
                "1",
            ]
        )
    finally:
        sys.stdout, sys.stderr = stdout, stderr

    assert exit_code != 0
    err = err_buf.getvalue()
    assert "ERROR" in err
    assert "suggested_domain='medical'" in err
    # Strict mode aborts before producing aggregates.
    assert "Hallucination-Guard benchmark" not in out_buf.getvalue()


def test_strict_domain_passes_when_matching(tmp_path: Path) -> None:
    dataset = tmp_path / "matched.json"
    dataset.write_text(
        json.dumps(
            {
                "suggested_domain": "general",
                "cases": [
                    {"query": "[EASY] anything?", "tag": "[EASY]"},
                ],
            }
        ),
        encoding="utf-8",
    )

    err_buf = io.StringIO()
    stderr = sys.stderr
    sys.stderr = err_buf
    try:
        with __import__("contextlib").redirect_stdout(io.StringIO()):
            exit_code = hallucination_rate.main(
                [
                    "--dataset",
                    str(dataset),
                    "--domain",
                    "general",
                    "--strict-domain",
                    "--max-retries",
                    "1",
                ]
            )
    finally:
        sys.stderr = stderr

    assert exit_code == 0
    assert "WARNING" not in err_buf.getvalue()
    assert "ERROR" not in err_buf.getvalue()


def test_simulator_emits_good_source_for_easy_tag() -> None:
    # The good_source param flows through to every "good" branch so the
    # citation matches whichever domain is configured.
    sim = hallucination_rate._SimulatedStructuredLLM(
        good_source="https://pubmed.ncbi.nlm.nih.gov/example/"
    )
    out = sim.generate(system="", user="[EASY] Q?", schema=object)  # type: ignore[arg-type]
    assert out.claims[0].sources == ["https://pubmed.ncbi.nlm.nih.gov/example/"]


def test_benchmark_async_matches_sync_aggregates() -> None:
    """The async path uses the same deterministic simulator wrapped in
    an awaitable shim; the aggregate numbers must match the sync run
    byte-for-byte modulo per-case ordering (which ``asyncio.gather``
    preserves)."""
    sync_buf = io.StringIO()
    async_buf = io.StringIO()
    stdout = sys.stdout
    try:
        sys.stdout = sync_buf
        sync_code = hallucination_rate.main(["--max-retries", "3"])
        sys.stdout = async_buf
        async_code = hallucination_rate.main(
            ["--max-retries", "3", "--async", "--concurrency", "4"]
        )
    finally:
        sys.stdout = stdout

    assert sync_code == 0
    assert async_code == 0
    # Strip the header lines that mention pid/timing if any were added;
    # for now both runs emit byte-identical aggregates.
    assert sync_buf.getvalue() == async_buf.getvalue()


def test_benchmark_async_concurrency_actually_overlaps() -> None:
    """With ``--concurrency >= 2`` and several cases, at least two cases
    must be observed running concurrently. Otherwise the async fan-out
    is silently degenerating into a serial loop."""
    domain_name = "general"
    sync_structured = hallucination_rate._SimulatedStructuredLLM(
        good_source=hallucination_rate._DOMAIN_GOOD_SOURCE[domain_name]
    )
    sync_judge = hallucination_rate._SimulatedJudgeLLM()
    async_structured = hallucination_rate._AsyncSimulatedStructuredLLM(
        sync_structured
    )
    async_judge = hallucination_rate._AsyncSimulatedJudgeLLM(sync_judge)

    from hallucination_guard.domain.general import GeneralDomain
    from hallucination_guard.graph import Graph

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=async_structured,
        judge_llm=async_judge,
        max_retries=3,
    )
    cases = tuple(
        hallucination_rate._Case(
            query=f"[EASY] Q{i}?", tag="[EASY]"
        )
        for i in range(8)
    )

    import asyncio as _asyncio

    outcomes = _asyncio.run(
        hallucination_rate._run_all_async(graph, cases, concurrency=4)
    )

    assert len(outcomes) == 8
    assert all(o.final.is_success for o in outcomes)
    assert async_structured.peak_in_flight >= 2
    assert async_structured.peak_in_flight <= 4


def test_benchmark_async_concurrency_one_is_serial() -> None:
    """``--concurrency 1`` must serialize cases — the peak in-flight
    counter should stay at 1 throughout."""
    sync_structured = hallucination_rate._SimulatedStructuredLLM(
        good_source=hallucination_rate._DOMAIN_GOOD_SOURCE["general"]
    )
    sync_judge = hallucination_rate._SimulatedJudgeLLM()
    async_structured = hallucination_rate._AsyncSimulatedStructuredLLM(
        sync_structured
    )
    async_judge = hallucination_rate._AsyncSimulatedJudgeLLM(sync_judge)

    from hallucination_guard.domain.general import GeneralDomain
    from hallucination_guard.graph import Graph

    graph = Graph(
        domain=GeneralDomain(),
        structured_llm=async_structured,
        judge_llm=async_judge,
        max_retries=3,
    )
    cases = tuple(
        hallucination_rate._Case(query=f"[EASY] Q{i}?", tag="[EASY]")
        for i in range(4)
    )

    import asyncio as _asyncio

    _asyncio.run(
        hallucination_rate._run_all_async(graph, cases, concurrency=1)
    )

    assert async_structured.peak_in_flight == 1


def test_benchmark_async_rejects_zero_concurrency() -> None:
    err_buf = io.StringIO()
    stderr = sys.stderr
    sys.stderr = err_buf
    try:
        with pytest.raises(SystemExit):
            hallucination_rate.main(["--async", "--concurrency", "0"])
    finally:
        sys.stderr = stderr
    assert "--concurrency must be a positive integer" in err_buf.getvalue()


def test_benchmark_async_real_requires_model() -> None:
    """The async branch shares the ``--real`` + ``--model`` requirement
    so the AsyncOpenAI adapters never silently default to a model name."""
    with pytest.raises(ValueError, match="--real requires --model"):
        hallucination_rate._build_graph(
            max_retries=1,
            use_real=True,
            real_model=None,
            domain_name="general",
            use_async=True,
        )
