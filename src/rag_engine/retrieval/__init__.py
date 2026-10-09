"""Retrieval: embedders, vector store, BM25 index, retrievers, and index persistence."""

from rag_engine.retrieval.bm25 import BM25Index, tokenize
from rag_engine.retrieval.embedders import Embedder, HashingEmbedder, l2_normalize
from rag_engine.retrieval.fastembed_embedder import FastEmbedEmbedder
from rag_engine.retrieval.index_store import embedder_from_spec, load_index, save_index
from rag_engine.retrieval.retriever import (
    DenseRetriever,
    LexicalRetriever,
    RetrievedChunk,
    Retriever,
)
from rag_engine.retrieval.vector_store import (
    Filters,
    InMemoryVectorStore,
    SearchHit,
    VectorStore,
    matches,
)

__all__ = [
    "BM25Index",
    "DenseRetriever",
    "Embedder",
    "FastEmbedEmbedder",
    "Filters",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "LexicalRetriever",
    "RetrievedChunk",
    "Retriever",
    "SearchHit",
    "VectorStore",
    "embedder_from_spec",
    "l2_normalize",
    "load_index",
    "matches",
    "save_index",
    "tokenize",
]
