"""Configurable chunking with exact character offsets and structural metadata."""

from collections.abc import Iterable

from rag_engine.chunking.base import Chunker, build_chunks
from rag_engine.chunking.stats import chunk_stats
from rag_engine.chunking.strategies import (
    FixedTokenChunker,
    RecursiveChunker,
    StructureAwareChunker,
    build_chunker,
)
from rag_engine.chunking.tokens import TokenIndex, count_tokens
from rag_engine.config import ChunkingConfig
from rag_engine.models import Chunk, Document


def chunk_documents(documents: Iterable[Document], config: ChunkingConfig) -> list[Chunk]:
    chunker = build_chunker(config)
    return [chunk for document in documents for chunk in chunker.chunk(document)]


__all__ = [
    "Chunker",
    "FixedTokenChunker",
    "RecursiveChunker",
    "StructureAwareChunker",
    "TokenIndex",
    "build_chunker",
    "build_chunks",
    "chunk_documents",
    "chunk_stats",
    "count_tokens",
]
