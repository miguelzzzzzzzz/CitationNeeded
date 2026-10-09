"""Retrieval primitives: embedders, vector stores, and the BM25 index."""

from rag_engine.retrieval.bm25 import BM25Index, tokenize
from rag_engine.retrieval.embedders import Embedder, HashingEmbedder, l2_normalize
from rag_engine.retrieval.fastembed_embedder import FastEmbedEmbedder
from rag_engine.retrieval.vector_store import (
    Filters,
    InMemoryVectorStore,
    SearchHit,
    VectorStore,
    matches,
)

__all__ = [
    "BM25Index",
    "Embedder",
    "FastEmbedEmbedder",
    "Filters",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "SearchHit",
    "VectorStore",
    "l2_normalize",
    "matches",
    "tokenize",
]
