"""Hallucination-rate benchmark for the framework.

Runs the full :class:`Graph` over a synthetic QA set loaded from a JSON
file and reports

- success rate (``is_success`` on the terminal :class:`GraphState`)
- average ``retry_count``
- breakdown of terminal ``fail_reason`` values

By default a deterministic *simulated* LLM pair is used so the benchmark can
run anywhere without API keys. The simulator decides each attempt's quality
from tags embedded in the query — this exercises every routing branch of
the graph and lets the numbers be reproduced exactly. Real LLMs are easy to
substitute by passing ``--real`` (requires ``OPENAI_API_KEY``).

The dataset is loaded from JSON so cases can be edited or replaced without
touching code. ``--dataset PATH`` selects a custom file; the default lives
at ``benchmarks/datasets/synthetic_qa.json``. A ``--domain`` flag selects
between :class:`GeneralDomain` (default) and :class:`MedicalDomain`; the
simulator picks a domain-valid citation URL accordingly so success cases
actually pass each domain's ``is_valid_source``.

Run me with:

    python -m benchmarks.hallucination_rate
    python -m benchmarks.hallucination_rate --verbose
    python -m benchmarks.hallucination_rate --max-retries 2
    python -m benchmarks.hallucination_rate --dataset path/to/cases.json
    python -m benchmarks.hallucination_rate --domain medical \\
        --dataset benchmarks/datasets/medical_qa.json
    python -m benchmarks.hallucination_rate --domain medical --strict-domain
    python -m benchmarks.hallucination_rate --async --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from hallucination_guard.domain.base import DomainConfig
from hallucination_guard.domain.general import GeneralDomain
from hallucination_guard.domain.medical import MedicalDomain
from hallucination_guard.graph import Graph
from hallucination_guard.schemas import Claim, CriticVerdict, GroundedOutput
from hallucination_guard.state import FailReason, GraphState


# Tags steer the simulator so we can hit every routing branch deterministically.
_TAG_EASY = "[EASY]"
_TAG_LOW_CONF = "[LOW_CONF]"
_TAG_BAD_SOURCE = "[BAD_SOURCE]"
_TAG_CRITIC_FAIL = "[CRITIC_FAIL]"
_TAG_RECOVERS = "[RECOVERS]"


_DEFAULT_DATASET = Path(__file__).parent / "datasets" / "synthetic_qa.json"

# Per-domain source URLs the simulator emits on "good" attempts. The values
# must satisfy the domain's ``is_valid_source`` so the deterministic numbers
# stay reproducible after a domain swap.
_DOMAIN_GOOD_SOURCE: dict[str, str] = {
    "general": "https://en.wikipedia.org/wiki/Example",
    "medical": "https://pubmed.ncbi.nlm.nih.gov/00000000/",
}


@dataclass(frozen=True)
class _Case:
    query: str
    tag: str


@dataclass(frozen=True)
class _Dataset:
    cases: tuple[_Case, ...]
    suggested_domain: str | None


def _load_dataset(path: Path) -> _Dataset:
    """Load cases and metadata from a JSON file.

    The optional top-level ``suggested_domain`` string is surfaced on the
    returned :class:`_Dataset` so callers can warn when a CLI ``--domain``
    selection contradicts the dataset's stated intent.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset {path} is not valid JSON: {exc}") from exc

    cases_raw = raw.get("cases") if isinstance(raw, dict) else None
    if not isinstance(cases_raw, list):
        raise ValueError(
            f"Dataset {path} is missing a 'cases' array at the top level"
        )

    cases: list[_Case] = []
    for i, entry in enumerate(cases_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset {path}: case #{i} is not an object")
        query = entry.get("query")
        tag = entry.get("tag", "")
        if not isinstance(query, str) or not query:
            raise ValueError(
                f"Dataset {path}: case #{i} is missing a non-empty 'query'"
            )
        if not isinstance(tag, str):
            raise ValueError(f"Dataset {path}: case #{i} 'tag' must be a string")
        cases.append(_Case(query=query, tag=tag))

    suggested = raw.get("suggested_domain") if isinstance(raw, dict) else None
    if suggested is not None and not isinstance(suggested, str):
        raise ValueError(
            f"Dataset {path}: 'suggested_domain' must be a string when present"
        )

    return _Dataset(cases=tuple(cases), suggested_domain=suggested)


class _SimulatedStructuredLLM:
    """Deterministic structured-output simulator keyed off query tags.

    ``good_source`` is the citation URL emitted on *successful* attempts and
    on the [LOW_CONF] / [CRITIC_FAIL] paths where the source itself is meant
    to be valid (the failure comes from elsewhere). [BAD_SOURCE] always
    emits ``http://`` regardless so the gate has something concrete to
    reject, independent of which domain is in play.
    """

    def __init__(
        self,
        *,
        good_source: str = "https://en.wikipedia.org/wiki/Example",
    ) -> None:
        self._good_source = good_source
        # Per-query attempt counter so [RECOVERS] can flip on the 2nd try.
        self._attempts: Counter[str] = Counter()

    def generate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        self._attempts[user] += 1
        attempt = self._attempts[user]

        if _TAG_LOW_CONF in user:
            return GroundedOutput(
                claims=[
                    Claim(
                        text="Unsure",
                        confidence=0.2,
                        sources=[self._good_source],
                    )
                ]
            )
        if _TAG_BAD_SOURCE in user:
            return GroundedOutput(
                claims=[
                    Claim(
                        text="Water boils at 100C",
                        confidence=0.95,
                        sources=["http://insecure.example/boil"],  # plain http
                    )
                ]
            )
        if _TAG_CRITIC_FAIL in user:
            return GroundedOutput(
                claims=[
                    Claim(
                        text="Contradiction",
                        confidence=0.96,
                        sources=[self._good_source],
                    )
                ]
            )
        if _TAG_RECOVERS in user and attempt == 1:
            return GroundedOutput(
                claims=[
                    Claim(
                        text="Tentative",
                        confidence=0.4,
                        sources=[self._good_source],
                    )
                ]
            )
        # _TAG_EASY or _TAG_RECOVERS on retry → confident, cited answer.
        return GroundedOutput(
            claims=[
                Claim(
                    text="Well-known fact",
                    confidence=0.95,
                    sources=[self._good_source],
                )
            ]
        )


class _SimulatedJudgeLLM:
    """Always PASS except when the structured layer emitted [CRITIC_FAIL]."""

    def judge(self, *, system: str, content: str) -> CriticVerdict:
        # The structured-layer payload for CRITIC_FAIL contains "Contradiction";
        # detect that to keep the simulator self-contained.
        if "Contradiction" in content:
            return CriticVerdict(
                verdict="FAIL",
                rejected_claims=["Contradiction"],
                reason="contradicts itself",
            )
        return CriticVerdict(verdict="PASS")


class _AsyncSimulatedStructuredLLM:
    """Async wrapper over :class:`_SimulatedStructuredLLM`.

    Yields once via ``await asyncio.sleep(0)`` so concurrent cases can
    actually interleave under the scheduler, then delegates to the sync
    simulator's deterministic logic. Tracking ``peak_in_flight`` exposes
    how many cases were simultaneously inside ``agenerate``, which the
    test suite uses to confirm the ``--concurrency`` bound is enforced.
    """

    def __init__(self, sync: _SimulatedStructuredLLM) -> None:
        self._sync = sync
        self.current_in_flight = 0
        self.peak_in_flight = 0

    async def agenerate(
        self, *, system: str, user: str, schema: type[BaseModel]
    ) -> BaseModel:
        self.current_in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.current_in_flight)
        try:
            await asyncio.sleep(0)
            return self._sync.generate(system=system, user=user, schema=schema)
        finally:
            self.current_in_flight -= 1


class _AsyncSimulatedJudgeLLM:
    """Async wrapper over :class:`_SimulatedJudgeLLM`."""

    def __init__(self, sync: _SimulatedJudgeLLM) -> None:
        self._sync = sync

    async def ajudge(self, *, system: str, content: str) -> CriticVerdict:
        await asyncio.sleep(0)
        return self._sync.judge(system=system, content=content)


@dataclass(frozen=True)
class _Outcome:
    query: str
    tag: str
    final: GraphState


def _run_all(graph: Graph, cases: Iterable[_Case]) -> list[_Outcome]:
    outcomes: list[_Outcome] = []
    for case in cases:
        result = graph.run(case.query)
        outcomes.append(_Outcome(query=case.query, tag=case.tag, final=result))
    return outcomes


async def _run_all_async(
    graph: Graph, cases: Iterable[_Case], concurrency: int
) -> list[_Outcome]:
    """Drive every case through ``graph.arun`` with a Semaphore-bounded fan-out.

    Each case becomes its own task; the semaphore caps how many are
    inside ``arun`` simultaneously. ``asyncio.gather`` preserves the
    input order in the returned list so the per-case verbose report
    keeps its dataset ordering regardless of completion order.
    """
    case_list = list(cases)
    sem = asyncio.Semaphore(concurrency)

    async def _one(case: _Case) -> _Outcome:
        async with sem:
            result = await graph.arun(case.query)
            return _Outcome(query=case.query, tag=case.tag, final=result)

    return list(await asyncio.gather(*(_one(c) for c in case_list)))


def _summarize(outcomes: list[_Outcome], max_retries: int) -> str:
    total = len(outcomes)
    if total == 0:
        return "(no cases)"

    successes = [o for o in outcomes if o.final.is_success]
    errors = [o for o in outcomes if not o.final.is_success]
    retry_counts = [o.final.retry_count for o in outcomes]
    success_retries = [o.final.retry_count for o in successes]

    failure_breakdown: Counter[str] = Counter()
    for o in errors:
        reason = o.final.fail_reason.value if o.final.fail_reason else "unknown"
        failure_breakdown[reason] += 1

    lines = [
        "Hallucination-Guard benchmark",
        "=============================",
        f"max_retries: {max_retries}",
        f"queries    : {total}",
        "",
        "Results:",
        f"  successes        : {len(successes):>3} / {total} "
        f"({len(successes) * 100 / total:.1f}%)",
        f"  errors           : {len(errors):>3} / {total} "
        f"({len(errors) * 100 / total:.1f}%)",
        f"  avg retry_count  : {statistics.fmean(retry_counts):.2f}",
    ]
    if success_retries:
        lines.append(
            f"  avg retries on success: {statistics.fmean(success_retries):.2f}"
        )
    if failure_breakdown:
        lines.append("")
        lines.append("Failure breakdown (terminal fail_reason):")
        for reason, count in sorted(failure_breakdown.items()):
            lines.append(f"  {reason:<18} : {count}")
    return "\n".join(lines)


def _per_case_table(outcomes: list[_Outcome]) -> str:
    lines = ["", "Per-query results:"]
    for o in outcomes:
        status = "PASS" if o.final.is_success else "FAIL"
        reason = ""
        if not o.final.is_success and o.final.fail_reason:
            reason = f"  reason={o.final.fail_reason.value}"
        lines.append(
            f"  [{status}] retries={o.final.retry_count}{reason}  {o.query}"
        )
    return "\n".join(lines)


def _build_domain(name: str) -> DomainConfig:
    if name == "general":
        return GeneralDomain()
    if name == "medical":
        return MedicalDomain()
    raise ValueError(f"Unknown domain: {name!r}")


def _build_graph(
    *,
    max_retries: int,
    use_real: bool,
    real_model: str | None,
    domain_name: str,
    use_async: bool = False,
) -> Graph:
    """Construct the benchmark :class:`Graph`.

    ``use_async=True`` picks async-flavour clients (simulator or
    ``AsyncOpenAI*Adapter``) so the resulting graph is in async mode and
    must be driven through :func:`_run_all_async`.
    """
    domain = _build_domain(domain_name)
    if use_real:
        if not real_model:
            raise ValueError("--real requires --model")
        # Imported here so the default path stays free of the openai dep.
        from hallucination_guard.llm.openai_adapter import (
            AsyncOpenAIJudgeAdapter,
            AsyncOpenAIStructuredAdapter,
            OpenAIJudgeAdapter,
            OpenAIStructuredAdapter,
        )

        if use_async:
            structured_llm: object = AsyncOpenAIStructuredAdapter(
                model=real_model
            )
            judge_llm: object = AsyncOpenAIJudgeAdapter(model=real_model)
        else:
            structured_llm = OpenAIStructuredAdapter(model=real_model)
            judge_llm = OpenAIJudgeAdapter(model=real_model)
    else:
        good_source = _DOMAIN_GOOD_SOURCE[domain_name]
        sync_structured = _SimulatedStructuredLLM(good_source=good_source)
        sync_judge = _SimulatedJudgeLLM()
        if use_async:
            structured_llm = _AsyncSimulatedStructuredLLM(sync_structured)
            judge_llm = _AsyncSimulatedJudgeLLM(sync_judge)
        else:
            structured_llm = sync_structured
            judge_llm = sync_judge

    return Graph(
        domain=domain,
        structured_llm=structured_llm,  # type: ignore[arg-type]
        judge_llm=judge_llm,  # type: ignore[arg-type]
        max_retries=max_retries,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per query (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-query results in addition to the aggregate.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help=f"Path to the QA JSON dataset (default: {_DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use the OpenAI adapter instead of the deterministic simulator "
        "(requires OPENAI_API_KEY and --model).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model identifier (required when --real is set).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=("general", "medical"),
        default="general",
        help="DomainConfig to instantiate (default: general).",
    )
    parser.add_argument(
        "--strict-domain",
        action="store_true",
        help="Treat a dataset/--domain mismatch as a hard error (exit != 0) "
        "instead of a stderr warning.",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Drive the benchmark through Graph.arun with concurrent fan-out "
        "(uses AsyncOpenAI adapters when combined with --real).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum cases in flight simultaneously when --async is set "
        "(default: 4). Ignored without --async.",
    )
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be a positive integer")

    dataset = _load_dataset(args.dataset)
    if (
        dataset.suggested_domain is not None
        and dataset.suggested_domain != args.domain
    ):
        level = "ERROR" if args.strict_domain else "WARNING"
        print(
            f"{level}: dataset {args.dataset} declares suggested_domain="
            f"{dataset.suggested_domain!r} but --domain is {args.domain!r}. "
            "Aggregate numbers may not match the dataset's intent.",
            file=sys.stderr,
        )
        if args.strict_domain:
            return 2
    graph = _build_graph(
        max_retries=args.max_retries,
        use_real=args.real,
        real_model=args.model,
        domain_name=args.domain,
        use_async=args.use_async,
    )
    if args.use_async:
        outcomes = asyncio.run(
            _run_all_async(graph, dataset.cases, args.concurrency)
        )
    else:
        outcomes = _run_all(graph, dataset.cases)

    print(_summarize(outcomes, max_retries=args.max_retries))
    if args.verbose:
        print(_per_case_table(outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
