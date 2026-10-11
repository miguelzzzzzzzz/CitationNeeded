"""Extractive, zero-cost answer generation (M5 default, ADR-0004).

Sentences with the highest query-term overlap are selected from the retrieved
chunks; no model calls, no network access, fully deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_engine.answering.models import AnswerResponse, citation_from_provenance
from rag_engine.retrieval.retriever import RetrievedChunk

__all__ = ["ExtractiveAnswerGenerator"]

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def _tokenize(text: str) -> frozenset[str]:
    """Casefold and keep alphanumeric tokens, dropping 1-char noise/stopwords."""
    return frozenset(
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if len(token) > 1 and token not in _STOPWORDS
    )


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation followed by whitespace."""
    return [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]


def _dedupe_key(sentence: str) -> str:
    """Casefold + collapse whitespace so near-identical sentences dedupe."""
    return _WHITESPACE_RE.sub(" ", sentence.casefold()).strip()


def _meta_str(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def _meta_page_end(metadata: dict[str, Any], page: int | None) -> int | None:
    value = metadata.get("page_end", page)
    return value if isinstance(value, int) else page


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One scored sentence plus the document-level span it occupies."""

    score: float
    rank: int
    offset: int
    sentence: str
    start_char: int
    end_char: int
    hit: RetrievedChunk


class ExtractiveAnswerGenerator:
    """Zero-cost extractive generator: query-term overlap over chunk sentences."""

    @property
    def name(self) -> str:
        return "extractive"

    def generate(
        self,
        query: str,
        hits: Sequence[RetrievedChunk],
        *,
        max_sentences: int = 5,
    ) -> AnswerResponse:
        """Answer ``query`` with the best-matching sentences from ``hits``.

        Raises ``ValueError`` when the query is blank/whitespace-only or when
        ``max_sentences`` is below 1. Stopword-only queries abstain rather than
        raise (no content terms to score against).
        """
        if not query.strip():
            raise ValueError("query must contain non-whitespace text")
        if max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")

        query_tokens = _tokenize(query)
        if not query_tokens or not hits:
            return AnswerResponse(
                query=query,
                answer="",
                citations=(),
                generator=self.name,
                model_name=None,
                abstained=True,
            )

        candidates: list[_Candidate] = []
        for hit in hits:
            chunk = hit.chunk
            # Per-chunk cursor so repeated sentences map to successive offsets.
            cursor = 0
            for offset, sentence in enumerate(_split_sentences(chunk.text)):
                local = chunk.text.find(sentence, cursor)
                if local < 0:  # pragma: no cover - defensive fallback
                    local = cursor
                cursor = local + len(sentence)
                overlap = len(query_tokens & _tokenize(sentence))
                if overlap == 0:
                    continue
                candidates.append(
                    _Candidate(
                        score=overlap / len(query_tokens),
                        rank=hit.rank,
                        offset=offset,
                        sentence=sentence,
                        start_char=chunk.start_char + local,
                        end_char=chunk.start_char + local + len(sentence),
                        hit=hit,
                    )
                )

        candidates.sort(key=lambda c: (-c.score, c.rank, c.offset))

        seen: set[str] = set()
        selected: list[_Candidate] = []
        for candidate in candidates:
            key = _dedupe_key(candidate.sentence)
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)
            if len(selected) >= max_sentences:
                break

        if not selected:
            return AnswerResponse(
                query=query,
                answer="",
                citations=(),
                generator=self.name,
                model_name=None,
                abstained=True,
            )

        citations = tuple(
            citation_from_provenance(
                doc_id=candidate.hit.chunk.doc_id,
                chunk_id=candidate.hit.chunk.chunk_id,
                quote=candidate.sentence,
                source=_meta_str(candidate.hit.chunk.metadata, "source"),
                content_hash=_meta_str(candidate.hit.chunk.metadata, "content_hash"),
                heading_path=candidate.hit.chunk.heading_path,
                page=candidate.hit.chunk.page,
                page_end=_meta_page_end(candidate.hit.chunk.metadata, candidate.hit.chunk.page),
                start_char=candidate.start_char,
                end_char=candidate.end_char,
            )
            for candidate in selected
        )
        return AnswerResponse(
            query=query,
            answer=" ".join(candidate.sentence for candidate in selected),
            citations=citations,
            generator=self.name,
            model_name=None,
            abstained=False,
        )
