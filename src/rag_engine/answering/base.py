"""Answer generator protocol for the answering stage (M5 / ADR-0004).

The protocol keeps generators swappable: the default extractive generator
composes sentences from retrieved chunks, while OpenAI-compatible generators
delegate to an external chat completion API. Both return the same
``AnswerResponse`` so callers do not depend on a concrete backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_engine.answering.models import AnswerResponse
from rag_engine.retrieval.retriever import RetrievedChunk

__all__ = ["AnswerGenerator"]


class AnswerGenerator(Protocol):
    """Produces an answer for a query from ranked retrieval hits."""

    @property
    def name(self) -> str:
        """Stable identifier for the generator, e.g. ``"extractive"``."""
        ...

    def generate(
        self,
        query: str,
        hits: Sequence[RetrievedChunk],
        *,
        max_sentences: int = 5,
    ) -> AnswerResponse:
        """Generate an answer from ``hits``.

        ``max_sentences`` bounds extractive output; LLM-backed generators may
        use it as a soft prompt constraint while still returning prose.
        """
        ...
