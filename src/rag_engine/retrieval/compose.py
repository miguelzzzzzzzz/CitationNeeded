"""Assemble the retriever stack shared by the CLI and the HTTP service.

``rag-engine search``, ``rag-engine answer`` and ``POST /search``/``POST /answer``
all serve the same modes over one loaded index: ``dense``, ``lexical``, or
``hybrid`` (dense + lexical fused), each optionally followed by a cross-encoder
reranking stage. Building the stack in one place keeps the wiring identical
across entry points; argument rules that depend on how options were *given*
(e.g. a flag that only makes sense in hybrid mode) stay with each caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from rag_engine.retrieval.fusion import FusionMethod, HybridRetriever
from rag_engine.retrieval.rerank import Reranker, RerankingRetriever
from rag_engine.retrieval.retriever import DenseRetriever, LexicalRetriever, Retriever

__all__ = ["RETRIEVAL_MODES", "RetrievalMode", "compose_retriever"]

RetrievalMode = Literal["dense", "lexical", "hybrid"]
RETRIEVAL_MODES: tuple[RetrievalMode, ...] = ("dense", "lexical", "hybrid")


def compose_retriever(
    dense: DenseRetriever,
    lexical: LexicalRetriever,
    *,
    mode: str,
    fusion: FusionMethod | str = FusionMethod.RRF,
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
    candidates: int = 50,
    reranker: Reranker | None = None,
) -> Retriever:
    """Return the retriever for ``mode``, wrapped in a reranking stage if given one.

    ``fusion``, ``weights`` and ``rrf_k`` only affect ``hybrid``; ``candidates``
    is the fusion pool and the reranker's budget. Invalid values raise
    ``ValueError`` from the constructors (or here, for an unknown mode) before
    any query runs.
    """
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}")
    retriever: Retriever
    if mode == "dense":
        retriever = dense
    elif mode == "lexical":
        retriever = lexical
    else:
        retriever = HybridRetriever(
            [dense, lexical],
            method=fusion,
            weights=weights,
            rrf_k=rrf_k,
            candidates=candidates,
        )
    if reranker is not None:
        retriever = RerankingRetriever(retriever, reranker, candidates=candidates)
    return retriever
