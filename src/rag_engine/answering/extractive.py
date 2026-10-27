"""Extractive, zero-cost answer generation (M5 default, ADR-0004).

Sentences with the highest query-term overlap are selected from the retrieved
chunks; no model calls, no network access, fully deterministic.

Markdown and HTML chunks keep their ATX heading lines (``# Title``) in the
chunk text (HTML headings are normalized to ATX at load time). For those
formats heading lines are sentence boundaries and never part of a quote.
Headings are found per chunk with the loader's fence-aware scanner, so a chunk
that starts inside a fenced code block can take ``#`` comment lines in that
block for headings and miss headings after its closing fence. Plain-text and
PDF chunks are split on punctuation only.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_engine.answering.models import AnswerResponse, citation_from_provenance
from rag_engine.ingestion.loaders import find_markdown_headings
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
# Formats whose normalized text carries ATX heading lines (see loaders.py).
_HEADING_FORMATS: frozenset[str] = frozenset({"markdown", "html"})
_WHITESPACE_RE = re.compile(r"\s+")


def _tokenize(text: str) -> frozenset[str]:
    """Casefold and keep alphanumeric tokens, dropping 1-char noise/stopwords."""
    return frozenset(
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if len(token) > 1 and token not in _STOPWORDS
    )


def _heading_line_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of every ATX heading line in ``text`` (end excludes the newline)."""
    spans: list[tuple[int, int]] = []
    for heading in find_markdown_headings(text):
        end = text.find("\n", heading.offset)
        spans.append((heading.offset, len(text) if end < 0 else end))
    return spans


def _sentence_spans(text: str, *, skip_headings: bool) -> list[tuple[int, int]]:
    """Exact ``(start, end)`` spans of the sentences in ``text``, stripped of whitespace.

    Sentences end at sentence-final punctuation followed by whitespace. With
    ``skip_headings`` every ATX heading line is also a boundary and is dropped,
    so no span contains a heading.
    """
    blocks: list[tuple[int, int]] = []
    cursor = 0
    if skip_headings:
        for start, end in _heading_line_spans(text):
            blocks.append((cursor, start))
            cursor = end
    blocks.append((cursor, len(text)))

    spans: list[tuple[int, int]] = []
    for block_start, block_end in blocks:
        piece_start = block_start
        block = text[block_start:block_end]
        bounds = [(m.start(), m.end()) for m in _SENTENCE_RE.finditer(block)]
        bounds.append((len(block), len(block)))
        for gap_start, gap_end in bounds:
            piece_end = block_start + gap_start
            piece = text[piece_start:piece_end]
            stripped = piece.strip()
            if stripped:
                lead = len(piece) - len(piece.lstrip())
                spans.append((piece_start + lead, piece_start + lead + len(stripped)))
            piece_start = block_start + gap_end
    return spans


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
            skip_headings = chunk.metadata.get("format") in _HEADING_FORMATS
            spans = _sentence_spans(chunk.text, skip_headings=skip_headings)
            for offset, (local, local_end) in enumerate(spans):
                sentence = chunk.text[local:local_end]
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
                        end_char=chunk.start_char + local_end,
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
