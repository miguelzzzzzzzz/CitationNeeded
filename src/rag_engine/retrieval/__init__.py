"""Retrieval primitives: embedders and vector stores."""

from rag_engine.retrieval.embedders import Embedder, HashingEmbedder, l2_normalize
from rag_engine.retrieval.vector_store import (
    Filters,
    InMemoryVectorStore,
    SearchHit,
    VectorStore,
    matches,
)

__all__ = [
    "Embedder",
    "Filters",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "SearchHit",
    "VectorStore",
    "l2_normalize",
    "matches",
]
