"""Hybrid fusion of several retrievers' ranked result lists.

Two strategies are provided. Reciprocal Rank Fusion scores a chunk by

    score(chunk) = sum over lists containing it of  weight[list] / (k + rank)

Only positions enter the formula, so RRF never has to compare score magnitudes:
cosine similarities and BM25 scores live on different scales, and making them
comparable (or calibrating a shared cut-off) would need corpus-specific tuning
that RRF simply does not require.

Weighted fusion first min-max normalizes each list to [0, 1] and then sums

    score(chunk) = sum over lists of  weight[list] * (s - min_list) / (max_list - min_list)

Normalization is per list *and per query*: the spread of scores a retriever
produces depends on the query (one query has a single dominant match, another a
flat distribution), so reusing a scale measured on a different query would
distort the configured weights. Chunks absent from a list contribute 0 for it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from rag_engine.models import Chunk
from rag_engine.retrieval.retriever import RetrievedChunk, Retriever
from rag_engine.retrieval.vector_store import Filters

__all__ = [
    "FusionMethod",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
]


class FusionMethod(StrEnum):
    """Hybrid fusion strategies accepted by :class:`HybridRetriever`."""

    RRF = "rrf"
    WEIGHTED = "weighted"


@dataclass
class _Candidate:
    """Mutable accumulator for one chunk while several lists are merged."""

    chunk: Chunk
    best_rank: int
    first_seen: int
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)


def _order_key(candidate: _Candidate) -> tuple[float, int, int]:
    """Score desc, then best original rank, then first appearance (stable ties)."""
    return (-candidate.score, candidate.best_rank, candidate.first_seen)


class _FusionAccumulator:
    """Collects per-chunk contributions and emits a deterministic ranked list.

    Chunks are keyed by ``chunk_id``; the first list (in mapping order) that
    contains a chunk supplies the payload and settles tie-breaks, so fusing the
    same inputs twice always yields the same output order.
    """

    def __init__(self) -> None:
        self._candidates: dict[str, _Candidate] = {}
        self._order = 0

    def add(self, name: str, hit: RetrievedChunk, contribution: float) -> None:
        """Add ``contribution`` to ``hit``'s fused score for result list ``name``."""
        chunk_id = hit.chunk.chunk_id
        candidate = self._candidates.get(chunk_id)
        if candidate is None:
            candidate = _Candidate(chunk=hit.chunk, best_rank=hit.rank, first_seen=self._order)
            self._order += 1
            self._candidates[chunk_id] = candidate
        elif hit.rank < candidate.best_rank:
            candidate.best_rank = hit.rank
        candidate.score += contribution
        candidate.components[name] = hit.score

    def finalize(self, retriever: str, top_k: int | None) -> list[RetrievedChunk]:
        """Sort, truncate to ``top_k`` and renumber ranks from 1."""
        ranked = sorted(self._candidates.values(), key=_order_key)
        if top_k is not None:
            ranked = ranked[:top_k]
        return [
            RetrievedChunk(
                chunk=candidate.chunk,
                score=candidate.score,
                rank=position,
                retriever=retriever,
                components=dict(candidate.components),
            )
            for position, candidate in enumerate(ranked, start=1)
        ]


def _dedupe(hits: Sequence[RetrievedChunk], name: str) -> list[RetrievedChunk]:
    """Reject a list repeating a chunk; a duplicate would double-count it."""
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for hit in hits:
        chunk_id = hit.chunk.chunk_id
        if chunk_id in seen:
            raise ValueError(f"result list {name!r} contains chunk {chunk_id!r} more than once")
        seen.add(chunk_id)
        unique.append(hit)
    return unique


def _validate_top_k(top_k: int | None) -> None:
    """``top_k`` is optional but, when given, must select at least one hit."""
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")


def _resolve_weights(names: Iterable[str], weights: Mapping[str, float] | None) -> dict[str, float]:
    """Return a weight per list; unknown names and negative weights are rejected.

    Omitted names default to 1.0 so callers can weight a single retriever
    without restating the rest of the configuration.
    """
    known = list(names)
    if weights is None:
        return dict.fromkeys(known, 1.0)
    unknown = sorted(set(weights) - set(known))
    if unknown:
        raise ValueError(f"weights reference unknown result lists: {unknown}")
    resolved: dict[str, float] = {}
    for name in known:
        weight = weights.get(name, 1.0)
        if weight < 0:
            raise ValueError(f"weight for {name!r} must be non-negative")
        resolved[name] = weight
    return resolved


def reciprocal_rank_fusion(
    results: Mapping[str, Sequence[RetrievedChunk]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    ``results`` maps a retriever name to its hits, best first, and each hit's
    ``rank`` field (1-based) supplies the position used in ``weight / (k + rank)``.
    The returned chunks carry ``retriever="hybrid-rrf"``, the fused score, fresh
    1-based ranks and ``components`` holding each list's original score.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    _validate_top_k(top_k)
    resolved = _resolve_weights(results.keys(), weights)
    accumulator = _FusionAccumulator()
    for name, hits in results.items():
        weight = resolved[name]
        for hit in _dedupe(hits, name):
            accumulator.add(name, hit, weight / (k + hit.rank))
    return accumulator.finalize("hybrid-rrf", top_k)


def weighted_score_fusion(
    results: Mapping[str, Sequence[RetrievedChunk]],
    *,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists by min-max normalizing each list, then weighting it.

    A list whose scores are all equal (including a single hit) normalizes every
    hit to 1.0, because there is no spread to rank within. The returned chunks
    carry ``retriever="hybrid-weighted"`` and ``components`` holding the original
    (unnormalized) per-list scores.
    """
    _validate_top_k(top_k)
    resolved = _resolve_weights(results.keys(), weights)
    accumulator = _FusionAccumulator()
    for name, hits in results.items():
        unique = _dedupe(hits, name)
        if not unique:
            continue
        weight = resolved[name]
        low = min(hit.score for hit in unique)
        high = max(hit.score for hit in unique)
        span = high - low
        for hit in unique:
            normalized = 1.0 if span == 0.0 else (hit.score - low) / span
            accumulator.add(name, hit, weight * normalized)
    return accumulator.finalize("hybrid-weighted", top_k)


class HybridRetriever:
    """Combines several retrievers by fusing their ranked lists.

    Every delegate is queried with the same text and filters and asked for
    ``max(k, candidates)`` hits, so fusion sees a pool deep enough to re-rank
    even when the caller wants only a handful of results, while metadata filters
    still apply inside each delegate before its own ranking is built.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        method: FusionMethod | str = FusionMethod.RRF,
        weights: Mapping[str, float] | None = None,
        rrf_k: int = 60,
        candidates: int = 50,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        names = [retriever.name for retriever in retrievers]
        if len(set(names)) != len(names):
            raise ValueError(f"retriever names must be unique, got {names}")
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        try:
            resolved_method = FusionMethod(method)
        except ValueError as exc:
            raise ValueError(f"unknown fusion method: {method!r}") from exc
        if weights is not None:
            unknown = sorted(set(weights) - set(names))
            if unknown:
                raise ValueError(f"weights reference unknown retrievers: {unknown}")
        self._retrievers = tuple(retrievers)
        self._method = resolved_method
        self._weights = None if weights is None else dict(weights)
        self._rrf_k = rrf_k
        self._candidates = candidates

    @property
    def name(self) -> str:
        return "hybrid"

    def retrieve(
        self, query: str, k: int = 10, filters: Filters | None = None
    ) -> list[RetrievedChunk]:
        """Query every delegate with identical filters, then fuse and truncate."""
        if k <= 0:
            raise ValueError("k must be positive")
        depth = max(k, self._candidates)
        results = {
            retriever.name: retriever.retrieve(query, depth, filters)
            for retriever in self._retrievers
        }
        if self._method is FusionMethod.RRF:
            return reciprocal_rank_fusion(results, k=self._rrf_k, weights=self._weights, top_k=k)
        return weighted_score_fusion(results, weights=self._weights, top_k=k)
