"""Retrieval: embedders, indexes, retrievers, hybrid fusion, reranking, persistence."""

from rag_engine.retrieval.bm25 import BM25Index, tokenize
from rag_engine.retrieval.embedders import Embedder, HashingEmbedder, l2_normalize
from rag_engine.retrieval.fastembed_embedder import FastEmbedEmbedder
from rag_engine.retrieval.fusion import (
    FusionMethod,
    HybridRetriever,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from rag_engine.retrieval.index_store import embedder_from_spec, load_index, save_index
from rag_engine.retrieval.rerank import CrossEncoderReranker, Reranker, RerankingRetriever
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
    "CrossEncoderReranker",
    "DenseRetriever",
    "Embedder",
    "FastEmbedEmbedder",
    "Filters",
    "FusionMethod",
    "HashingEmbedder",
    "HybridRetriever",
    "InMemoryVectorStore",
    "LexicalRetriever",
    "Reranker",
    "RerankingRetriever",
    "RetrievedChunk",
    "Retriever",
    "SearchHit",
    "VectorStore",
    "embedder_from_spec",
    "l2_normalize",
    "load_index",
    "matches",
    "reciprocal_rank_fusion",
    "save_index",
    "tokenize",
    "weighted_score_fusion",
]
