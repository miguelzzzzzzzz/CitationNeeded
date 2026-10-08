"""Chunking strategies.

* ``FixedTokenChunker``: sliding window of ``chunk_size`` tokens with
  ``chunk_overlap`` tokens of overlap, ignoring document structure. Baseline.
* ``RecursiveChunker``: splits on paragraph, line, sentence, then word
  boundaries until pieces fit, then packs pieces greedily with overlap.
* ``StructureAwareChunker``: applies the recursive strategy *within* each
  section (Markdown/HTML heading or PDF page) so chunks never straddle
  section boundaries unless a tiny section is merged into its neighbour.

All strategies return spans into ``Document.text``; ``build_chunks`` turns
them into chunks, so offsets are exact for every strategy.
"""

from __future__ import annotations

import re
from itertools import pairwise

from rag_engine.chunking.base import Span, build_chunks
from rag_engine.chunking.tokens import TokenIndex
from rag_engine.config import ChunkingConfig, ChunkStrategy
from rag_engine.models import Chunk, Document

# Ordered from coarsest to finest; a split point is the *end* of a match so the
# separator stays attached to the preceding piece.
SEPARATORS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n\s*\n"),  # paragraphs
    re.compile(r"\n"),  # lines
    re.compile(r"(?<=[.!?])[\"')\]]?\s+"),  # sentences
    re.compile(r"\s+"),  # words
)


def _token_windows(tokens: TokenIndex, start: int, end: int, size: int, overlap: int) -> list[Span]:
    first = tokens.first_at_or_after(start)
    last = tokens.first_at_or_after(end)  # exclusive
    if first >= last:
        return []
    step = size - overlap
    spans: list[Span] = []
    for window_start in range(first, last, step):
        window_end = min(window_start + size, last)
        spans.append((tokens.starts[window_start], tokens.ends[window_end - 1]))
        if window_end == last:
            break
    return spans


class FixedTokenChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = TokenIndex(document.text)
        spans = _token_windows(
            tokens, 0, len(document.text), self.config.chunk_size, self.config.chunk_overlap
        )
        return build_chunks(document, spans, self.config)


class RecursiveChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = TokenIndex(document.text)
        spans = self.split_range(document.text, tokens, 0, len(document.text))
        return build_chunks(document, spans, self.config)

    def split_range(self, text: str, tokens: TokenIndex, start: int, end: int) -> list[Span]:
        pieces = self._atomic_pieces(text, tokens, start, end, 0)
        return self._pack(pieces, tokens)

    def _atomic_pieces(
        self, text: str, tokens: TokenIndex, start: int, end: int, level: int
    ) -> list[Span]:
        """Split ``[start, end)`` into pieces of at most ``chunk_size`` tokens."""
        if tokens.count(start, end) <= self.config.chunk_size:
            return [(start, end)]
        if level >= len(SEPARATORS):
            return _token_windows(tokens, start, end, self.config.chunk_size, 0)
        cuts = [
            m.end() for m in SEPARATORS[level].finditer(text, start, end) if start < m.end() < end
        ]
        if not cuts:
            return self._atomic_pieces(text, tokens, start, end, level + 1)
        pieces: list[Span] = []
        bounds = [start, *cuts, end]
        for piece_start, piece_end in pairwise(bounds):
            pieces.extend(self._atomic_pieces(text, tokens, piece_start, piece_end, level + 1))
        return pieces

    def _pack(self, pieces: list[Span], tokens: TokenIndex) -> list[Span]:
        """Greedily pack pieces into chunks, carrying trailing pieces as overlap."""
        size, overlap = self.config.chunk_size, self.config.chunk_overlap
        counts = [tokens.count(s, e) for s, e in pieces]
        spans: list[Span] = []
        i, n = 0, len(pieces)
        while i < n:
            j, total = i, 0
            while j < n and total + counts[j] <= size:
                total += counts[j]
                j += 1
            j = max(j, i + 1)  # pieces never exceed size, but guarantee progress
            spans.append((pieces[i][0], pieces[j - 1][1]))
            if j >= n:
                break
            # Step back over trailing pieces to form the overlap, keeping room for
            # at least the next unseen piece so every chunk adds new content.
            k, carried = j, 0
            while (
                k - 1 > i
                and carried + counts[k - 1] <= overlap
                and carried + counts[k - 1] + counts[j] <= size
            ):
                carried += counts[k - 1]
                k -= 1
            i = k
        return spans


class StructureAwareChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        self._recursive = RecursiveChunker(config)

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = TokenIndex(document.text)
        spans: list[Span] = []
        for section in document.sections:
            spans.extend(
                self._recursive.split_range(
                    document.text, tokens, section.start_char, section.end_char
                )
            )
        return build_chunks(document, spans, self.config)


def build_chunker(
    config: ChunkingConfig,
) -> FixedTokenChunker | RecursiveChunker | StructureAwareChunker:
    if config.strategy is ChunkStrategy.FIXED:
        return FixedTokenChunker(config)
    if config.strategy is ChunkStrategy.RECURSIVE:
        return RecursiveChunker(config)
    return StructureAwareChunker(config)
