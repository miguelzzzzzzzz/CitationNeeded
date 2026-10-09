"""Tests for :class:`rag_engine.retrieval.fusion.HybridRetriever`.

The two fusion functions and the dense/lexical retrievers have test modules of
their own; everything here drives the hybrid retriever through the ``Retriever``
protocol, either with tiny fakes (so the fused arithmetic can be checked by hand)
or with the real in-process retrievers (uniqueness and filter propagation).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from rag_engine.models import Chunk
from rag_engine.retrieval.embedders import HashingEmbedder
from rag_engine.retrieval.fusion import (
    FusionMethod,
    HybridRetriever,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from rag_engine.retrieval.retriever import (
    DenseRetriever,
    LexicalRetriever,
    RetrievedChunk,
)
from rag_engine.retrieval.vector_store import Filters


def make_chunk(
    chunk_id: str,
    *,
    text: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Chunk:
    """Build a valid chunk; text and metadata fall back to deterministic values."""
    body = f"text for {chunk_id}" if text is None else text
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        index=0,
        text=body,
        start_char=0,
        end_char=len(body),
        token_count=len(body.split()),
        metadata={} if metadata is None else dict(metadata),
    )


def hit(chunk_id: str, rank: int, score: float, retriever: str) -> RetrievedChunk:
    """Build one result exactly as a single retriever would report it."""
    return RetrievedChunk(chunk=make_chunk(chunk_id), score=score, rank=rank, retriever=retriever)


class FakeRetriever:
    """Records every call and replays a fixed, pre-ranked result list."""

    def __init__(self, name: str, hits: Sequence[RetrievedChunk]) -> None:
        self._name = name
        self._hits = list(hits)
        self.calls: list[tuple[str, int, Filters | None]] = []

    @property
    def name(self) -> str:
        return self._name

    def retrieve(
        self, query: str, k: int = 10, filters: Filters | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((query, k, filters))
        return self._hits[:k]


def _fake(name: str) -> FakeRetriever:
    """A one-hit retriever used where only call plumbing matters."""
    return FakeRetriever(name, [hit("a", 1, 1.0, name)])


def _dense_and_lexical() -> tuple[FakeRetriever, FakeRetriever]:
    """dense = [a, b, c] and lexical = [c, a, d], with scores on different scales."""
    dense = FakeRetriever(
        "dense",
        [hit("a", 1, 0.9, "dense"), hit("b", 2, 0.6, "dense"), hit("c", 3, 0.3, "dense")],
    )
    lexical = FakeRetriever(
        "lexical",
        [hit("c", 1, 7.0, "lexical"), hit("a", 2, 3.0, "lexical"), hit("d", 3, 1.0, "lexical")],
    )
    return dense, lexical


def test_retrieve_queries_every_delegate_with_shared_depth_and_filters() -> None:
    dense = FakeRetriever("dense", [hit("a", 1, 0.5, "dense")])
    lexical = FakeRetriever("lexical", [hit("b", 1, 0.5, "lexical")])
    filters: Filters = {"topic": "math"}
    hybrid = HybridRetriever([dense, lexical], candidates=2)

    results = hybrid.retrieve("neural search", k=5, filters=filters)

    # depth = max(k=5, candidates=2) = 5, and both delegates get the same query
    assert dense.calls == [("neural search", 5, filters)]
    assert lexical.calls == [("neural search", 5, filters)]
    # the very same filters object is forwarded, not a defensive copy
    assert dense.calls[0][2] is filters
    assert lexical.calls[0][2] is filters
    # both chunks score 1/61; the tie falls back to the first list and first hit
    assert [result.chunk.chunk_id for result in results] == ["a", "b"]


def test_retrieve_asks_for_candidates_when_it_exceeds_k() -> None:
    dense = FakeRetriever("dense", [hit("a", 1, 0.5, "dense")])
    hybrid = HybridRetriever([dense], candidates=10)

    hybrid.retrieve("query", k=3)

    # depth = max(k=3, candidates=10) = 10, and no filters were passed
    assert dense.calls == [("query", 10, None)]


def test_retrieve_orders_results_by_fused_rrf_score() -> None:
    dense, lexical = _dense_and_lexical()
    hybrid = HybridRetriever([dense, lexical])

    results = hybrid.retrieve("query", k=4)

    # k=60; rank r contributes 1/(60+r):
    #   a = 1/61 + 1/62 = 0.0325224...
    #   b = 1/62        = 0.0161290...
    #   c = 1/63 + 1/61 = 0.0322664...
    #   d = 1/63        = 0.0158730...
    # so a > c > b > d
    assert [result.chunk.chunk_id for result in results] == ["a", "c", "b", "d"]
    assert [result.rank for result in results] == [1, 2, 3, 4]
    assert all(result.retriever == "hybrid-rrf" for result in results)
    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert results[1].score == pytest.approx(1 / 63 + 1 / 61)
    assert results[2].score == pytest.approx(1 / 62)
    assert results[3].score == pytest.approx(1 / 63)
    # components keep each list's original score, not the fused one
    assert results[0].components == pytest.approx({"dense": 0.9, "lexical": 3.0})
    assert results[2].components == pytest.approx({"dense": 0.6})
    assert results[3].components == pytest.approx({"lexical": 1.0})


def test_retrieve_truncates_to_top_k_after_fusing() -> None:
    dense, lexical = _dense_and_lexical()
    # candidates defaults to 50, so both delegates still contribute all three hits
    hybrid = HybridRetriever([dense, lexical])

    results = hybrid.retrieve("query", k=2)

    # truncation keeps the full fused scores: a = 1/61 + 1/62, c = 1/63 + 1/61
    assert [result.chunk.chunk_id for result in results] == ["a", "c"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert results[1].score == pytest.approx(1 / 63 + 1 / 61)


def test_retrieve_honours_configured_rrf_k() -> None:
    dense, lexical = _dense_and_lexical()
    hybrid = HybridRetriever([dense, lexical], method=FusionMethod.RRF, rrf_k=0)

    results = hybrid.retrieve("query", k=4)

    # with k=0, rank r contributes 1/r:
    #   a = 1/1 + 1/2 = 1.5
    #   b = 1/2       = 0.5
    #   c = 1/3 + 1/1 = 1.3333...
    #   d = 1/3       = 0.3333...
    assert [result.chunk.chunk_id for result in results] == ["a", "c", "b", "d"]
    assert results[0].score == pytest.approx(1.5)
    assert results[1].score == pytest.approx(1.0 + 1 / 3)
    assert results[2].score == pytest.approx(0.5)
    assert results[3].score == pytest.approx(1 / 3)


def test_retrieve_with_weighted_method_reports_hybrid_weighted() -> None:
    dense, lexical = _dense_and_lexical()
    hybrid = HybridRetriever(
        [dense, lexical], method="weighted", weights={"dense": 1.0, "lexical": 2.0}
    )

    results = hybrid.retrieve("query", k=4)

    # dense: min=0.3 span=0.6 -> a=(0.9-0.3)/0.6=1.0, b=(0.6-0.3)/0.6=0.5, c=0.0
    # lexical: min=1.0 span=6.0 -> c=(7-1)/6=1.0, a=(3-1)/6=1/3, d=0.0
    # a = 1.0*1.0 + 2.0*(1/3) = 1.6666...
    # b = 1.0*0.5 = 0.5
    # c = 1.0*0.0 + 2.0*1.0 = 2.0
    # d = 2.0*0.0 = 0.0
    assert [result.chunk.chunk_id for result in results] == ["c", "a", "b", "d"]
    assert all(result.retriever == "hybrid-weighted" for result in results)
    assert results[0].score == pytest.approx(2.0)
    assert results[1].score == pytest.approx(1.0 + 2.0 * (1 / 3))
    assert results[2].score == pytest.approx(0.5)
    assert results[3].score == pytest.approx(0.0)
    # components stay unnormalized: the raw per-list scores
    assert results[0].components == pytest.approx({"dense": 0.3, "lexical": 7.0})


_INVALID_CONFIGS: list[tuple[list[str], dict[str, Any], str]] = [
    ([], {}, "at least one retriever"),
    (["dense", "dense"], {}, "retriever names must be unique"),
    (["dense"], {"candidates": 0}, "candidates must be at least 1"),
    (["dense"], {"method": "bogus"}, "unknown fusion method"),
    (["dense"], {"weights": {"sparse": 1.0}}, "weights reference unknown retrievers"),
]


@pytest.mark.parametrize(("names", "kwargs", "match"), _INVALID_CONFIGS)
def test_constructor_rejects_invalid_configuration(
    names: list[str], kwargs: dict[str, Any], match: str
) -> None:
    retrievers = [_fake(name) for name in names]

    with pytest.raises(ValueError, match=match):
        HybridRetriever(retrievers, **kwargs)


def test_retrieve_rejects_non_positive_k_without_calling_delegates() -> None:
    dense = FakeRetriever("dense", [hit("a", 1, 0.5, "dense")])
    hybrid = HybridRetriever([dense])

    for bad_k in (0, -3):
        with pytest.raises(ValueError, match="k must be positive"):
            hybrid.retrieve("query", k=bad_k)

    assert dense.calls == []


def test_hybrid_retrieves_each_chunk_once_and_honours_metadata_filters() -> None:
    chunks = [
        make_chunk(
            "c1",
            text="vector search finds the nearest neighbours quickly",
            metadata={"topic": "math", "source": "a.md"},
        ),
        make_chunk(
            "c2",
            text="vector index stores embeddings for similarity",
            metadata={"topic": "math", "source": "b.md"},
        ),
        make_chunk(
            "c3",
            text="lexical search matches the exact query terms",
            metadata={"topic": "math", "source": "c.md"},
        ),
        make_chunk(
            "c4",
            text="bm25 ranks documents by term frequency",
            metadata={"topic": "math", "source": "d.md"},
        ),
        make_chunk(
            "c5",
            text="vector vector vector soup recipes for the winter",
            metadata={"topic": "cooking", "source": "e.md"},
        ),
    ]
    dense = DenseRetriever(HashingEmbedder())
    lexical = LexicalRetriever()
    dense.index(chunks)
    lexical.index(chunks)
    hybrid = HybridRetriever([dense, lexical], candidates=5)
    filters: Filters = {"topic": "math"}

    results = hybrid.retrieve("vector search similarity", k=5, filters=filters)

    ids = [result.chunk.chunk_id for result in results]
    assert ids, "expected at least one fused result"
    # every chunk id appears at most once, and ranks are renumbered from 1
    assert len(ids) == len(set(ids))
    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    assert all(result.retriever == "hybrid-rrf" for result in results)
    # the filter applies inside both delegates: c5 is out despite matching "vector"
    assert set(ids) <= {"c1", "c2", "c3", "c4"}
    assert all(result.chunk.metadata.get("topic") == "math" for result in results)


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_rrf_rejects_non_finite_scores(score: float) -> None:
    results = {"dense": [hit("a", 1, score, "dense")]}

    with pytest.raises(ValueError, match="non-finite score"):
        reciprocal_rank_fusion(results)


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_weighted_fusion_rejects_non_finite_scores(score: float) -> None:
    results = {
        "dense": [hit("a", 1, 0.5, "dense")],
        "lexical": [hit("b", 1, score, "lexical")],
    }

    with pytest.raises(ValueError, match="non-finite score"):
        weighted_score_fusion(results)


def test_rrf_rejects_rank_below_one() -> None:
    results = {"dense": [hit("a", 0, 1.0, "dense")]}

    with pytest.raises(ValueError, match="rank below 1"):
        reciprocal_rank_fusion(results)


def test_accepts_generator_of_retrievers() -> None:
    dense, lexical = _dense_and_lexical()

    hybrid = HybridRetriever(r for r in (dense, lexical))
    hits = hybrid.retrieve("q")

    assert len(dense.calls) == 1
    assert len(lexical.calls) == 1
    assert {h.chunk.chunk_id for h in hits} == {"a", "b", "c", "d"}
    assert {h.retriever for h in hits} == {"hybrid-rrf"}


def test_rejects_negative_rrf_k() -> None:
    with pytest.raises(ValueError, match="rrf_k must be non-negative"):
        HybridRetriever([_fake("dense")], rrf_k=-1)


@pytest.mark.parametrize("weight", [-1.0, float("nan")])
def test_rejects_invalid_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        HybridRetriever([_fake("dense")], weights={"dense": weight})


def test_rejects_all_zero_weights() -> None:
    retrievers = [_fake("dense"), _fake("lexical")]

    with pytest.raises(ValueError, match="at least one weight must be positive"):
        HybridRetriever(
            retrievers,
            method="weighted",
            weights={"dense": 0.0, "lexical": 0.0},
        )
