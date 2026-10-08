"""Chunker protocol and the shared span -> Chunk assembly logic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_engine.chunking.tokens import TokenIndex
from rag_engine.config import ChunkingConfig
from rag_engine.models import Chunk, Document, sha256_hex

Span = tuple[int, int]


class Chunker(Protocol):
    config: ChunkingConfig

    def chunk(self, document: Document) -> list[Chunk]: ...


def trim_span(text: str, start: int, end: int) -> Span:
    """Shrink ``[start, end)`` so it neither starts nor ends with whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _common_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    prefix: list[str] = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        prefix.append(x)
    return tuple(prefix)


def merge_small_spans(
    spans: Sequence[Span], tokens: TokenIndex, config: ChunkingConfig
) -> list[Span]:
    """Merge spans below ``min_chunk_tokens`` into a contiguous neighbour.

    A small span is appended to the previous span (or prepended to the next)
    only when the merged span still fits in ``chunk_size``; otherwise it is
    kept as is. Merged spans cover the text between them, so the offset
    invariant still holds.
    """
    if config.min_chunk_tokens == 0 or len(spans) < 2:
        return list(spans)
    merged: list[Span] = []
    pending: Span | None = None  # small span waiting to be prepended to the next one
    for start, end in spans:
        if pending is not None:
            if tokens.count(pending[0], end) <= config.chunk_size:
                start = pending[0]
            else:
                merged.append(pending)
            pending = None
        size = tokens.count(start, end)
        if size >= config.min_chunk_tokens:
            merged.append((start, end))
            continue
        if merged and tokens.count(merged[-1][0], end) <= config.chunk_size:
            merged[-1] = (merged[-1][0], end)
        else:
            pending = (start, end)
    if pending is not None:
        merged.append(pending)
    return merged


def build_chunks(document: Document, spans: Sequence[Span], config: ChunkingConfig) -> list[Chunk]:
    """Turn character spans into ``Chunk`` objects with provenance metadata."""
    tokens = TokenIndex(document.text)
    trimmed = [trim_span(document.text, s, e) for s, e in spans]
    non_empty = [(s, e) for s, e in trimmed if e > s]
    final = merge_small_spans(non_empty, tokens, config)

    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(final):
        first = document.section_at(start)
        last = document.section_at(end - 1)
        heading_path: tuple[str, ...] = ()
        if first is not None and last is not None:
            heading_path = _common_prefix(first.heading_path, last.heading_path)
        metadata: dict[str, object] = {
            "source": document.source,
            "title": document.title,
            "format": document.format,
            **document.metadata,
        }
        page = first.page if first is not None else None
        if last is not None and last.page is not None and last.page != page:
            metadata["page_end"] = last.page
        header = ""
        if config.include_heading_context:
            parts = [document.title, *heading_path]
            if len(parts) > 1 and parts[0] == parts[1]:
                parts = parts[1:]
            header = " > ".join(p for p in parts if p)
        chunk_id = sha256_hex(f"{document.doc_id}:{document.content_hash}:{start}:{end}")[:16]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                index=index,
                text=document.text[start:end],
                start_char=start,
                end_char=end,
                token_count=tokens.count(start, end),
                heading_path=heading_path,
                page=page,
                context_header=header,
                metadata=metadata,
            )
        )
    return chunks
