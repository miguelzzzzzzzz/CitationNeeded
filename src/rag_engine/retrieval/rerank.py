"""Cross-encoder reranking of first-stage retrieval results (milestone M3).

A cross-encoder scores every ``(query, chunk)`` pair jointly, which is far more
accurate than comparing independently embedded vectors and correspondingly more
expensive: scoring cost grows linearly with the number of pairs. The pipeline is
therefore always retrieve-then-rerank: :class:`RerankingRetriever` pulls
``max(k, candidates)`` hits from its base retriever, scores exactly that many
pairs in one model call, and returns the top ``k``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol

from rag_engine.retrieval.retriever import RetrievedChunk, Retriever
from rag_engine.retrieval.vector_store import Filters

__all__ = [
    "CrossEncoderModel",
    "CrossEncoderReranker",
    "Reranker",
    "RerankingRetriever",
]


class CrossEncoderModel(Protocol):
    """The subset of ``fastembed.TextCrossEncoder`` this adapter uses."""

    def rerank(
        self, query: str, documents: Iterable[str], batch_size: int = ...
    ) -> Iterable[float]: ...


class Reranker(Protocol):
    """Scores how relevant each text is to a query (higher = more relevant)."""

    @property
    def name(self) -> str: ...

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """``Reranker`` backed by a fastembed ``TextCrossEncoder``.

    ``fastembed`` is an optional dependency and is imported in :meth:`_load`, so
    importing this module (and the default test suite) never requires it.
    Passing ``model`` injects a stand-in, which keeps reranking tests offline.
    """

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        *,
        batch_size: int = 32,
        cache_dir: str | None = None,
        threads: int | None = None,
        model: CrossEncoderModel | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.threads = threads
        self._model = model

    @property
    def name(self) -> str:
        return f"cross-encoder:{self.model_name}"

    def _load(self) -> CrossEncoderModel:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on install
                raise ImportError(
                    "fastembed is not installed; run: pip install -e '.[embeddings]'"
                ) from exc
            self._model = TextCrossEncoder(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=self.threads,
            )
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            # Nothing to score: skip the lazy load (and any model download).
            return []
        model = self._load()
        scores = [
            float(value) for value in model.rerank(query, list(texts), batch_size=self.batch_size)
        ]
        if len(scores) != len(texts):
            raise RuntimeError(f"reranker returned {len(scores)} scores for {len(texts)} texts")
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError("reranker returned NaN or infinite scores")
        return scores


class RerankingRetriever:
    """Two-stage retrieval: a wide first stage, a cross-encoder second stage.

    Cost model: the reranker sees at most ``max(k, candidates)`` ``(query,
    chunk)`` pairs per query, so ``candidates`` is the latency budget knob --
    raising it costs strictly linearly in scoring time and reduces the chance
    that the best chunk never reaches the reranker. ``k`` may exceed
    ``candidates``, which is why the base retriever is asked for
    ``max(k, candidates)`` hits.
    """

    def __init__(self, base: Retriever, reranker: Reranker, *, candidates: int = 50) -> None:
        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        self._base = base
        self._reranker = reranker
        self._candidates = candidates

    @property
    def name(self) -> str:
        return f"{self._base.name}+rerank"

    @property
    def base(self) -> Retriever:
        return self._base

    @property
    def reranker(self) -> Reranker:
        return self._reranker

    def retrieve(
        self, query: str, k: int = 10, filters: Filters | None = None
    ) -> list[RetrievedChunk]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not query.strip():
            # No query signal; the base retrievers short-circuit the same way.
            return []
        hits = self._base.retrieve(query, max(k, self._candidates), filters)
        if not hits:
            return []
        scores = self._reranker.score(query, [hit.chunk.embedding_text for hit in hits])
        if len(scores) != len(hits):
            raise RuntimeError(
                f"reranker {self._reranker.name} returned {len(scores)} scores "
                f"for {len(hits)} candidates"
            )
        pairs = list(zip(hits, scores, strict=True))
        # Descending reranker score; the explicit base rank breaks ties
        # deterministically, independent of the base retriever's list order.
        pairs.sort(key=lambda item: (-item[1], item[0].rank))
        stage = self._base.name
        results: list[RetrievedChunk] = []
        for position, (hit, score) in enumerate(pairs[:k], start=1):
            # A stage keeps its own score and rank slot, so reranking a reranker
            # extends the record instead of clobbering the inner stage's data.
            if stage in hit.components or stage in hit.ranks:
                raise ValueError(
                    f"stage name {stage!r} already recorded for chunk "
                    f"{hit.chunk.chunk_id!r}; nested stages need distinct names"
                )
            results.append(
                RetrievedChunk(
                    chunk=hit.chunk,
                    score=score,
                    rank=position,
                    retriever=self.name,
                    components={**hit.components, stage: hit.score},
                    ranks={**hit.ranks, stage: hit.rank},
                )
            )
        return results
