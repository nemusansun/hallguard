# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.2] — 2026-05-12

### Changed

- **PyPI distribution renamed from `hallucination-guard` to `hallguard`.**
  Install with `pip install hallguard` from PyPI. The Python import name
  is unchanged: `import hallucination_guard` and
  `from hallucination_guard.graph import Graph` continue to work, so
  no source-level changes are required for consumers who depend on the
  import name. Only the distribution name on PyPI (and therefore the
  `pip install ...` argument) is different.

### Why

The original distribution name `hallucination-guard` was claimed on
production PyPI by an unrelated project on 2025-08-16 before this
project's first publish attempt. PEP 541 name claims have effectively
no chance of succeeding against an actively-published namesake, so
the distribution was renamed rather than blocked indefinitely. The
0.8.1 build was successfully uploaded to TestPyPI under the old name
during this session as a final verification before the conflict was
discovered on production upload, then superseded by this 0.8.2 build
under the new name.

## [0.8.1] — 2026-05-12

### Added

- **PyPI distribution metadata.** `pyproject.toml` now sets
  `description`, `readme = "README.md"`, `license = "MIT"`,
  `license-files = ["LICENSE"]`, `authors`, `keywords`, and a
  full `classifiers` list (Development Status, Intended Audience,
  Python 3.11/3.12/3.13, AI / Libraries topics, `Typing :: Typed`).
  Before this change `twine check` warned that
  `long_description` was missing, which would have shipped a
  README-less project page on PyPI.
- **`LICENSE` file (MIT)** at the repository root, referenced by
  `license-files` so the wheel and sdist both carry it.
- **`setuptools>=77`** in `[build-system].requires` so the build
  environment understands PEP 639 SPDX `license` expressions.
- **High-concurrency `arun` stress tests**
  (`test_arun_high_concurrency_matches_serial_results`,
  `test_arun_concurrent_calls_do_not_share_fail_history`) fan out 64
  fake-LLM queries through a single `Graph` under
  `asyncio.Semaphore(32)` and assert per-query results match a serial
  baseline, that `fail_history` never leaks from failing into
  succeeding concurrent calls, and that the simulator's
  `peak_in_flight` counter confirms real overlap (`>= 2`) within the
  semaphore bound (`<= 32`).
- **Locale-aware retry instructions.** `RetryHintBuilder.INSTRUCTION_MAPS`
  is now a `dict[Locale, dict[FailReason, str]]` with both `"en"` and
  `"ja"` phrasings; `RetryHintBuilder.build(state, *, locale="en")`
  picks the right entry. `DomainConfig.retry_locale()` (concrete,
  default `"en"`) lets domains forward their locale to the builder;
  `GeneralDomain` / `MedicalDomain` override it to return their
  constructor `locale`. `StructuredNode` plumbs the value through so
  retry hints stay in the same language as the rest of the prompt.

### Fixed

- **Package `__version__` now matches the distribution version.**
  `hallucination_guard.__version__` had been stuck at `"0.1.0"` since
  the initial scaffolding and is now `"0.8.1"`, so
  `import hallucination_guard; hallucination_guard.__version__` no
  longer disagrees with `pyproject.toml::version` after install.
- **`locale="en"` retry prompts are now actually English.** Previously
  the hint builder's `INSTRUCTION_MAP` was hard-coded Japanese, so a
  domain configured with `locale="en"` would inject a Japanese
  `fix_instruction` line into its otherwise English retry directive.
  See the locale-aware INSTRUCTION_MAPS change above for the fix.
- **README per-file test counts now match the suite.** The detailed
  table previously summed to 182 while the headline said 185, because
  several files had drifted in either direction since their counts
  were last edited. The table now reflects the actual tallies and the
  headline reads 193.

### Changed

- README test counts updated to reflect the new totals (183 → 193);
  `tests/test_graph_integration.py` row notes the high-concurrency
  state-isolation stress tests, and `tests/test_hint_builder.py` plus
  the domain test rows mention the new locale coverage.

## [0.8.0] — 2026-05-12

### Added

- **`--async` / `--concurrency N`** on
  `benchmarks/hallucination_rate.py`. The benchmark CLI can now drive
  the full dataset through `Graph.arun` with `asyncio.gather` and a
  Semaphore-bounded fan-out. Combined with `--real --model <id>` it
  picks up the new `AsyncOpenAI*Adapter`s.
- **`_AsyncSimulatedStructuredLLM` / `_AsyncSimulatedJudgeLLM`**
  wrappers in the benchmark module. They delegate to the existing
  deterministic simulators after yielding to the event loop, and
  expose a `peak_in_flight` counter so tests can prove the
  concurrency bound is actually enforced.
- **Cancellation / timeout tests** for `Graph.arun` and `Graph.astream`
  (`asyncio.wait_for` timeout, explicit `task.cancel()`, mid-stream
  consumer cancel, post-completion `task.cancel()` is a no-op).

### Changed

- `benchmarks/hallucination_rate._build_graph` accepts
  `use_async=True` and picks the appropriate sync vs async client
  pair (simulator or OpenAI adapter).
- README test counts updated to reflect the new totals (174 → 183).

### Migration notes

- No breaking changes. Existing benchmark invocations without
  `--async` keep their previous behavior and produce byte-identical
  aggregates.
- The aggregate numbers from `--async` are guaranteed to match the
  sync run on the bundled deterministic simulator; a regression test
  (`test_benchmark_async_matches_sync_aggregates`) enforces this.

## [0.7.0] — 2026-05-12

### Added

- **`AsyncStructuredLLM` / `AsyncJudgeLLM` protocols** in
  `hallucination_guard.llm.protocols`. Both are `@runtime_checkable`
  and expose distinct method names (`agenerate` / `ajudge`) from their
  sync counterparts so a single adapter class can implement both
  surfaces without method collisions.
- **`AsyncOpenAIStructuredAdapter` / `AsyncOpenAIJudgeAdapter`** in
  `hallucination_guard.llm.openai_adapter`. Built around `AsyncOpenAI`
  so calls can be awaited inside an event loop without blocking.
- **`Graph.arun()`** — async single-shot counterpart of `Graph.run()`.
  Works regardless of whether the configured clients are sync or async,
  giving callers already inside an event loop a uniform async surface.
- **`Graph.is_async`** read-only property. Reflects whether the graph
  was constructed with at least one async-only client.
- **Async-native pipeline path.** `Graph` now inspects its LLM
  arguments at construction time and, if either is async-only, wraps
  nodes through an async LangGraph runner. `StructuredNode` / `CriticNode`
  gained `acall()` methods that await `agenerate` / `ajudge` when
  available and fall back to the sync surface otherwise — letting a
  single graph host both async LLM nodes and CPU-only nodes without
  ad-hoc bridging.

### Changed

- `Graph.run()` and `Graph.stream()` now raise `RuntimeError` with a
  pointer to `arun()` / `astream()` when the graph was configured with
  an async-only client, instead of silently handing back coroutines
  via LangGraph.
- `examples/research_agent.py` demo 5 now uses the new
  `AsyncStructuredLLM` / `AsyncJudgeLLM` protocols directly through
  `Graph.arun()`, dropping the previous background-loop bridge.

### Migration notes

- No breaking changes for existing sync callers. Graphs constructed
  with sync clients keep their full sync API (`run` / `stream`) plus
  the previously-available `astream`, and now also expose `arun`.
- If you were using a custom sync-to-async bridge to call async LLM
  clients through `Graph.run`, you can now drop the bridge: implement
  `AsyncStructuredLLM` / `AsyncJudgeLLM` directly and call
  `await graph.arun(query)` (or iterate `graph.astream(query)`).
- Classes that happen to expose **both** `generate` and `agenerate`
  are treated as **sync** by `Graph` so existing `run()` callers see
  no behavior change. To opt into async dispatch, expose only the
  async method.

## [0.6.0] — 2026-05-12

### Added

- **`hallucination_guard.domain.base.Locale`** type alias
  (`Literal["en", "ja"]`) exposed for type-checking domain `locale`
  arguments.
- **`GeneralDomain(locale="ja")` / `MedicalDomain(locale="ja")`** — the
  built-in domains now accept a `locale` keyword and ship Japanese
  variants of their `system_prompt`, `critic_prompt`, and
  `format_retry_directive` strings. The default remains `"en"`, so
  existing callers see no behavior change. Verdict markers
  (`verdict=PASS` / `verdict=FAIL`) and proper-noun source brand names
  (PubMed, WHO, CDC, Cochrane, NEJM) stay ASCII even in the Japanese
  variant so they continue to match the host allow-list and the
  structured-output parser.
- **`_ALLOWED_HOSTS` ↔ retry-template integrity tests** in
  `tests/test_medical_domain.py`. Drift between the medical retry
  directive's "(PubMed, WHO, CDC, Cochrane, NEJM)" enumeration and the
  actual host allow-list now fails CI in either direction.
- **Async-LLM bridge demo** in `examples/research_agent.py`
  (`demo 5`). Shows an `async`-only client (mirroring vendor SDKs that
  expose only awaitable methods) bridged into the synchronous
  `StructuredLLM` protocol via a dedicated background event loop and
  driven through `Graph.astream`.

### Changed

- `GeneralDomain.__init__` and `MedicalDomain.__init__` now accept a
  keyword-only `locale: Locale = "en"`. Positional construction
  (`GeneralDomain()`, `MedicalDomain()`) is unchanged.

### Migration notes

- No breaking changes from `0.5.0`. The `Locale` default keeps every
  existing `GeneralDomain()` / `MedicalDomain()` call returning the
  same English prompts. Custom `DomainConfig` subclasses are
  unaffected — `locale` is not enforced by the abstract base; opt in
  by accepting your own `locale` kwarg.

## [0.5.0] — 2026-05-12

### Added

- **`DomainConfig.format_retry_directive(base_prompt, directive)`** abstract
  method. Domains now own the retry-prompt assembly so per-attempt wording —
  separators, forbidden-claims phrasing, language, etc. — stays out of the
  framework core. `GeneralDomain` keeps the existing English template;
  `MedicalDomain` ships a medical-flavored variant that names the allow-listed
  citation hosts and uses a stricter forbidden-claims preamble.
- **`--strict-domain` flag** on the benchmark CLI. Promotes the
  dataset/`--domain` mismatch warning to an error: the run aborts before
  producing aggregates and exits with a non-zero status. Without the flag,
  the previous warning-only behavior is preserved.
- **`Graph.astream()` demo** in `examples/research_agent.py` (`demo 4`).
  Drives the async streaming API via `asyncio.run` and mirrors the
  per-node event sequence already shown by `demo 3` (the synchronous
  `Graph.stream`).

### Changed

- `StructuredNode._with_directive` removed; the node now delegates retry
  prompt assembly to `domain.format_retry_directive(...)`. Behavior for
  `GeneralDomain` is unchanged.

### Migration notes

- Any `DomainConfig` subclass written against `≤0.4.0` must implement
  `format_retry_directive(base_prompt, directive)` — otherwise instantiation
  will raise `TypeError` from Python's abstract-method machinery. Reusing
  the previous behavior is a one-liner: copy `GeneralDomain.format_retry_directive`
  into your subclass and adjust the wording as needed.
- `StructuredNode._with_directive` is gone. Tests or tooling that monkey-patched
  it should instead override the relevant domain's `format_retry_directive`.

## [0.4.0] — 2026-05-12

### Added

- **`Graph(auto_serialize=True)`** opt-in flag that swaps the supplied
  checkpointer's default `JsonPlusSerializer` for one allow-listing this
  framework's Pydantic types — equivalent to constructing the checkpointer
  with `serde=build_serializer()` yourself. Refuses to overwrite a
  customized serializer so user-supplied allow-lists are never silently
  clobbered.
- **`hallucination_guard.serde.install_framework_serializer(checkpointer)`**
  exposes the swap logic for callers that build checkpointers
  independently. Same safety rails as the `Graph` flag.
- **`Graph.astream(query, *, thread_id=None)`** asynchronous counterpart
  of `Graph.stream`. Yields the identical `StreamEvent` sequence and
  preserves Pydantic instances in `research_output`; differs only in that
  awaiting node coroutines no longer blocks the surrounding event loop.
- **Dataset / `--domain` mismatch warning** on the benchmark CLI. The
  bundled `synthetic_qa.json` / `medical_qa.json` now declare
  `suggested_domain`; running with a different `--domain` prints a
  `WARNING:` line to stderr (the run still proceeds — only the
  expected aggregate numbers change).

### Changed

- `benchmarks.hallucination_rate._load_dataset(path)` now returns a
  `_Dataset(cases, suggested_domain)` dataclass instead of a bare tuple
  of cases. Bundled datasets ship a `suggested_domain` metadata field
  (`general` / `medical`); the field stays optional.

### Migration notes

- No breaking changes from `0.3.0` for `Graph.run` / `Graph.stream` /
  `Graph.get_state` callers. `auto_serialize` defaults to `False`, so
  pre-existing checkpointer wiring is unaffected.
- Custom benchmark drivers calling `_load_dataset` directly need to
  read `.cases` (and optionally `.suggested_domain`) on the returned
  value instead of treating it as a tuple.

## [0.3.0] — 2026-05-12

### Added

- **`Graph.stream(query, *, thread_id=None)`** yields a `StreamEvent`
  after every node executes. `StreamEvent.state` carries the cumulative
  `GraphState` (research_output keeps its concrete Pydantic instance, the
  same invariant `Graph.run` relies on) so progress UIs can react to each
  step without losing access to the final outcome.
- **`hallucination_guard.serde.build_serializer()`** returns a
  `JsonPlusSerializer` preloaded with this framework's Pydantic types so
  LangGraph 1.1+ stops emitting `Deserializing unregistered type ...
  This will be blocked in a future version.` warnings on checkpoint
  reads. Accepts extra classes for callers that ship custom
  `GroundedOutput` subclasses. A lower-level
  `framework_msgpack_modules()` is also exported for callers that want
  to compose the allow-list manually.
- **MedicalDomain benchmark dataset** at
  `benchmarks/datasets/medical_qa.json` plus a `--domain {general,medical}`
  switch on the benchmark CLI. The deterministic simulator now takes a
  `good_source` URL so successful attempts cite a host that passes each
  domain's `is_valid_source` — both domains produce identical aggregate
  numbers from the bundled datasets.
- **`Makefile`** with `check` / `build` / `release-check` / `release-test`
  / `release` targets wrapping the standard `python -m build` + `twine`
  flow. Documented in the README's new "リリース手順" section.
- **Streaming demo** in `examples/research_agent.py` — a third scenario
  prints per-node events through `Graph.stream`.

### Changed

- `_SimulatedStructuredLLM` accepts a `good_source` keyword so the
  citation URL emitted on the [EASY] / [LOW_CONF] / [CRITIC_FAIL] /
  [RECOVERS] branches matches the configured domain. [BAD_SOURCE]
  continues to emit `http://...` regardless so the gate has something
  unambiguously bad to reject.
- The [CRITIC_FAIL] confidence is now `0.96` (was `0.9`) so the case
  reaches the critic under any domain — `GeneralDomain`'s 0.7 threshold
  still passes and `MedicalDomain`'s 0.95 threshold no longer
  short-circuits to a gate failure.

### Migration notes

- No breaking changes from `0.2.0`. `Graph.run`, `Graph.get_state`,
  every existing node, and the public `DomainConfig` API are unchanged.
- Existing checkpointer users get the runtime warning silenced by
  switching `InMemorySaver()` → `InMemorySaver(serde=build_serializer())`
  — the old form still works, just keeps emitting the LangGraph warning.

## [0.2.0] — 2026-05-11

### Added

- **`DomainConfig.system_prompt()`** abstract method. Each domain now controls
  the system prompt handed to the structured-output LLM. `GeneralDomain` and
  `MedicalDomain` ship default implementations tuned to their respective
  source policies.
- **OpenAI adapter** (`hallucination_guard.llm.openai_adapter`) implementing
  `StructuredLLM` and `JudgeLLM` on top of `chat.completions.parse` with
  `response_format=<Pydantic class>` and `temperature=0`. The OpenAI client is
  injectable so tests can swap in an in-memory fake. The `model` keyword is
  required — no default is provided to avoid silently calling an unintended
  model.
- **`MedicalDomain`** (`hallucination_guard.domain.medical`) — demo strict
  domain enforcing `confidence>=0.95` and a small allow-list of medical
  source hosts (PubMed, WHO, CDC, Cochrane, NEJM).
- **Hallucination-rate benchmark** (`benchmarks.hallucination_rate`) with a
  deterministic simulated-LLM pair that exercises every routing branch and an
  optional `--real --model <id>` switch for live OpenAI runs.
- **JSON-backed benchmark dataset** at `benchmarks/datasets/synthetic_qa.json`
  loaded by the benchmark CLI; pass `--dataset PATH` to point at another file.
- **LangGraph checkpointer support** on `Graph`. Pass a `checkpointer` (e.g.
  `langgraph.checkpoint.memory.InMemorySaver`) and call `Graph.run(query,
  thread_id="...")`. Persisted state is retrievable with `Graph.get_state`.
- **`max_retries=0` edge-case tests** covering the “initial failure → straight
  to ErrorOutput” path on both the gate and critic branches.

### Changed

- **Core dependencies trimmed to `pydantic`.** `langgraph` and `openai` are
  now declared under `[project.optional-dependencies]` as the `graph`,
  `openai`, and aggregating `all` extras. Install with
  `pip install -e ".[all]"` (or `.[dev]` to add test tooling) to get the
  previous behavior. The dev extra additionally pins `mypy`.
- **`python-dotenv` dropped** from declared dependencies — it was never
  imported.
- **`StructuredNode`** no longer hardcodes the system prompt; it pulls one
  from `domain.system_prompt()` each call.

### Removed

- **`python-dotenv`** from `[project.dependencies]` (see Changed).

### Migration notes

- Any `DomainConfig` subclass written against `0.1.0` must implement
  `system_prompt()` — otherwise instantiation will raise `TypeError` from
  Python's abstract-method machinery.
- Users importing `Graph` (or the OpenAI adapter) must reinstall with the
  appropriate extra: `pip install -e ".[graph]"` for `Graph`,
  `pip install -e ".[openai]"` for the adapter, or `.[all]` for both.
- Adapter callers must pass `model=<id>` explicitly — there is no default.

## [0.1.0] — 2026-05-11

Initial portfolio release.

### Added

- `GraphState` / `FailReason` (immutable updates via `with_update`).
- `DomainConfig` abstract base plus a permissive `GeneralDomain`.
- Prompt-injection-safe retry layer: `RetryDirective` (frozen) and
  `RetryHintBuilder` (instructions sourced from a fixed `INSTRUCTION_MAP`).
- Nodes: `StructuredNode`, `FactCheckGate`, `CriticNode`, `RetryNode`,
  `ErrorOutput`.
- `Graph` wiring the nodes together via LangGraph with bounded retry
  routing (`retry_count >= max_retries` checked on every FAIL branch).
- `examples/research_agent.py` demonstrating the success and exhaustion
  paths against mock LLMs.
- Pytest suite covering state, hint builder, every node, the general
  domain, and the graph end-to-end.
