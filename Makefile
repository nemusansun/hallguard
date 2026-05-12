# Makefile for hallguard
#
# Common workflows for development, quality gates, and PyPI releases. All
# targets assume a local virtualenv at ./.venv; override PY if you keep
# Python somewhere else.

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest
MYPY ?= .venv/bin/mypy

PACKAGE := hallucination_guard

.DEFAULT_GOAL := help

.PHONY: help venv install install-dev test typecheck lint policy-check check \
        clean build release-test release release-check

help:  ## Show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv:  ## Create the virtualenv at ./.venv
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install:  ## Install the package in editable mode (no extras)
	$(PIP) install -e .

install-dev:  ## Install with the dev extra (langgraph + openai + pytest + mypy)
	$(PIP) install -e ".[dev]"
	$(PIP) install --upgrade build twine

test:  ## Run the test suite
	$(PYTEST) tests/ -q

typecheck:  ## Run mypy across source, benchmarks, and examples
	$(MYPY) $(PACKAGE)/ benchmarks/ examples/

policy-check:  ## Verify no AI/authorship markers leak into shipped files
	@! grep -rn -iE "claude|anthropic|chatgpt|copilot|gpt-|initial_prompt|session_log" \
	    $(PACKAGE) tests README.md pyproject.toml examples benchmarks CHANGELOG.md LICENSE \
	    || (echo "policy grep found banned strings" && exit 1)

check: test typecheck policy-check  ## Run every quality gate
	@echo "all checks passed"

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info $(PACKAGE).egg-info

build: clean  ## Build sdist + wheel into dist/
	$(PY) -m build

release-check: check build  ## Run quality gates then twine check the artifacts
	$(PY) -m twine check dist/*

release-test: release-check  ## Upload to TestPyPI (requires ~/.pypirc or env vars)
	$(PY) -m twine upload --repository testpypi dist/*

release: release-check  ## Upload to PyPI (requires ~/.pypirc or env vars)
	$(PY) -m twine upload dist/*
