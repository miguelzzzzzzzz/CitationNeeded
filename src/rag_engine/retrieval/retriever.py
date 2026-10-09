"""Retriever implementations over the in-process vector and BM25 indexes.

Retrievers share one contract (:class:`Retriever`) and one result shape
(:class:`RetrievedChunk`) so callers can fuse or compare dense and lexical
results without branching on the backend. ``RetrievedChunk.provenance`` is the
single source of truth for citation fields, keeping the retrieval layer's
output stable for downstream persistence and answering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rag_engine.models import Chunk
from rag_engine.retrieval.bm25 import BM25Index
from rag_engine.retrieval.embedders import Embedder
from rag_engine.retrieval.vector_store import Filters, InMemoryVectorStore

__all__ = ["DenseRetriever", "LexicalRetriever", "RetrievedChunk", "Retriever"]


@dataclass(frozen=True)
class RetrievedChunk:
    """A scored chunk plus the identity of the retriever that produced it.

    ``rank`` is the 1-based position within one retriever's result list; it is
    not comparable across retrievers (fusion assigns its own ordering).
    """

    chunk: Chunk
    score: float
    rank: int
    retriever: str

    def provenance(self) -> dict[str, Any]:
        """Citation fields shared by every retriever's output."""
        return {
            "doc_id": self.chunk.doc_id,
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.metadata.get("source"),
            "heading_path": list(self.chunk.heading_path),
            "page": self.chunk.page,
            "start_char": self.chunk.start_char,
            "end_char": self.chunk.end_char,
        }

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        """JSON-serializable view; key order matches the documented schema."""
        payload: dict[str, Any] = {
            "rank": self.rank,
            "score": self.score,
            "retriever": self.retriever,
        }
        payload.update(self.provenance())
        if include_text:
            payload["text"] = self.chunk.text
        return payload


class Retriever(Protocol):
    """Common interface for dense and lexical retrieval."""

    @property
    def name(self) -> str: ...

    def retrieve(
        self, query: str, k: int = 10, filters: Filters | None = None
    ) -> list[RetrievedChunk]: ...


def _validate_k(k: int) -> None:
    """Reject non-positive ``k`` before any backend work is attempted."""
    if k <= 0:
        raise ValueError("k must be positive")


class DenseRetriever:
    """Embedding-based retrieval against an :class:`InMemoryVectorStore`.

    The store is created from the embedder's dimension when not supplied; a
    caller-provided store must agree on dimension, otherwise upserts and
    searches would fail later with a less actionable error.
    """

    def __init__(self, embedder: Embedder, store: InMemoryVectorStore | None = None) -> None:
        if store is None:
            store = InMemoryVectorStore(embedder.dimension)
        elif store.dimension != embedder.dimension:
            raise ValueError(
                f"store dimension {store.dimension} does not match embedder dimension "
                f"{embedder.dimension}"
            )
        self._embedder = embedder
        self._store = store

    @property
    def name(self) -> str:
        return "dense"

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def store(self) -> InMemoryVectorStore:
        return self._store

    def index(self, chunks: Sequence[Chunk]) -> int:
        """Embed and upsert chunks; returns how many were processed (empty input: no embed)."""
        if not chunks:
            return 0
        vectors = self._embedder.embed_documents([chunk.embedding_text for chunk in chunks])
        self._store.upsert(chunks, vectors)
        return len(chunks)

    def retrieve(
        self, query: str, k: int = 10, filters: Filters | None = None
    ) -> list[RetrievedChunk]:
        _validate_k(k)
        if not query.strip():
            # A blank query has no signal; skip the (possibly remote) embed call.
            return []
        vector = self._embedder.embed_query(query)
        if not np.any(vector):
            # A zero query vector (e.g. no hashed features) scores every chunk 0;
            # returning that arbitrary list would look like a real ranking.
            return []
        hits = self._store.search(vector, k, filters)
        return [
            RetrievedChunk(chunk=hit.chunk, score=hit.score, rank=hit.rank, retriever=self.name)
            for hit in hits
        ]


class LexicalRetriever:
    """BM25 retrieval over an injected or freshly built :class:`BM25Index`."""

    def __init__(self, index: BM25Index | None = None) -> None:
        self._index = BM25Index() if index is None else index

    @property
    def name(self) -> str:
        return "lexical"

    @property
    def bm25(self) -> BM25Index:
        return self._index

    def index(self, chunks: Sequence[Chunk]) -> int:
        """Upsert chunks into the inverted index; returns how many chunks were processed."""
        self._index.upsert(chunks)
        return len(chunks)

    def retrieve(
        self, query: str, k: int = 10, filters: Filters | None = None
    ) -> list[RetrievedChunk]:
        _validate_k(k)
        if not query.strip():
            # Tokenizing whitespace yields no terms; short-circuit for symmetry
            # with the dense retriever and to avoid needless work.
            return []
        hits = self._index.search(query, k, filters)
        return [
            RetrievedChunk(chunk=hit.chunk, score=hit.score, rank=hit.rank, retriever=self.name)
            for hit in hits
        ]
