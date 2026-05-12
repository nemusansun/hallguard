"""Custom exception hierarchy for the framework.

All framework-raised errors derive from :class:`GraphError`, so callers can
``except GraphError`` to catch any failure originating inside the graph.
"""

from __future__ import annotations


class GraphError(Exception):
    """Base class for all errors raised inside the graph."""


class StructuredOutputError(GraphError):
    """The LLM returned output that did not conform to the declared schema."""


class CriticError(GraphError):
    """The critic LLM produced an unusable verdict."""
