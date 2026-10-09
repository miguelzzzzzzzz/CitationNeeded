"""Tests for weighted min-max fusion in :mod:`rag_engine.retrieval.fusion`."""

from __future__ import annotations

import pytest

from rag_engine.models import Chunk
from rag_engine.retrieval.fusion import weighted_score_fusion
from rag_engine.retrieval.retriever import RetrievedChunk


def make_chunk(chunk_id: str) -> Chunk:
    """Build a minimal valid chunk; fusion itself only ever touches ``chunk_id``."""
    text = f"content of {chunk_id}"
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
        metadata={"source": f"{chunk_id}.md"},
    )


def hit(chunk_id: str, rank: int, score: float, retriever: str) -> RetrievedChunk:
    """Build one retriever's ranked hit without running a real retriever."""
    return RetrievedChunk(chunk=make_chunk(chunk_id), score=score, rank=rank, retriever=retriever)


def test_normalizes_each_list_then_sums_weights() -> None:
    dense = [hit("a", 1, 0.9, "dense"), hit("b", 2, 0.5, "dense"), hit("c", 3, 0.1, "dense")]
    lexical = [hit("c", 1, 8.0, "lexical"), hit("a", 2, 2.0, "lexical")]

    fused = weighted_score_fusion({"dense": dense, "lexical": lexical})

    # dense span = 0.9 - 0.1 = 0.8 -> a = 0.8/0.8 = 1.0, b = 0.4/0.8 = 0.5, c = 0.0/0.8 = 0.0
    # lexical span = 8.0 - 2.0 = 6.0 -> c = 6.0/6.0 = 1.0, a = 0.0/6.0 = 0.0
    # weights default to 1.0: a = 1.0 + 0.0 = 1.0, c = 0.0 + 1.0 = 1.0, b = 0.5
    # the a/c tie at 1.0 breaks on best rank (both 1), then on first appearance (a wins)
    assert [r.chunk.chunk_id for r in fused] == ["a", "c", "b"]
    assert [r.score for r in fused] == pytest.approx([1.0, 1.0, 0.5])
    assert [r.rank for r in fused] == [1, 2, 3]
    assert [r.retriever for r in fused] == ["hybrid-weighted", "hybrid-weighted", "hybrid-weighted"]


def test_weights_scale_each_lists_normalized_contribution() -> None:
    dense = [hit("a", 1, 0.9, "dense"), hit("b", 2, 0.5, "dense"), hit("c", 3, 0.1, "dense")]
    lexical = [hit("c", 1, 8.0, "lexical"), hit("a", 2, 2.0, "lexical")]

    fused = weighted_score_fusion(
        {"dense": dense, "lexical": lexical}, weights={"dense": 2.0, "lexical": 0.5}
    )

    # dense normalizes to a = 1.0, b = 0.5, c = 0.0 and is weighted by 2.0
    # lexical normalizes to c = 1.0, a = 0.0 and is weighted by 0.5
    # a = 2.0*1.0 + 0.5*0.0 = 2.0 ; b = 2.0*0.5 = 1.0 ; c = 2.0*0.0 + 0.5*1.0 = 0.5
    assert [r.chunk.chunk_id for r in fused] == ["a", "b", "c"]
    assert [r.score for r in fused] == pytest.approx([2.0, 1.0, 0.5])


def test_chunk_absent_from_a_list_contributes_zero() -> None:
    dense = [hit("a", 1, 0.9, "dense"), hit("b", 2, 0.1, "dense")]
    lexical = [hit("c", 1, 8.0, "lexical"), hit("d", 2, 4.0, "lexical")]

    fused = weighted_score_fusion({"dense": dense, "lexical": lexical})

    # dense span = 0.8 -> a = 1.0, b = 0.0 ; lexical span = 4.0 -> c = 1.0, d = 0.0
    # b is absent from lexical and d is absent from dense, so each scores only its own list:
    # a = 1.0, c = 1.0, b = 0.0, d = 0.0
    assert [r.chunk.chunk_id for r in fused] == ["a", "c", "b", "d"]
    assert [r.score for r in fused] == pytest.approx([1.0, 1.0, 0.0, 0.0])
    # a chunk missing from a list carries no component for it at all
    assert fused[1].components == pytest.approx({"lexical": 8.0})


def test_identical_scores_in_a_list_normalize_to_one() -> None:
    dense = [hit("a", 1, 0.5, "dense"), hit("b", 2, 0.5, "dense"), hit("c", 3, 0.5, "dense")]

    fused = weighted_score_fusion({"dense": dense})

    # span = 0.5 - 0.5 = 0.0 -> no spread to rank within, so every hit becomes 1.0
    # order falls back to the original ranks: a (rank 1), b (rank 2), c (rank 3)
    assert [r.chunk.chunk_id for r in fused] == ["a", "b", "c"]
    assert [r.score for r in fused] == pytest.approx([1.0, 1.0, 1.0])


def test_single_hit_list_normalizes_to_one() -> None:
    fused = weighted_score_fusion(
        {
            "dense": [hit("a", 1, 0.7, "dense")],
            "lexical": [hit("c", 1, 8.0, "lexical"), hit("d", 2, 4.0, "lexical")],
        }
    )

    # the single-hit dense list has span 0.0 -> a = 1.0 (not 0.0, the raw score is ignored)
    # lexical span = 8.0 - 4.0 = 4.0 -> c = 4.0/4.0 = 1.0, d = 0.0/4.0 = 0.0
    # a = 1.0, c = 1.0 + 0.0 = 1.0, d = 0.0
    assert [r.chunk.chunk_id for r in fused] == ["a", "c", "d"]
    assert [r.score for r in fused] == pytest.approx([1.0, 1.0, 0.0])


def test_components_keep_the_original_unnormalized_scores() -> None:
    dense = [hit("a", 1, 0.9, "dense"), hit("b", 2, 0.5, "dense"), hit("c", 3, 0.1, "dense")]
    lexical = [hit("c", 1, 8.0, "lexical"), hit("a", 2, 2.0, "lexical")]

    fused = weighted_score_fusion(
        {"dense": dense, "lexical": lexical}, weights={"dense": 2.0, "lexical": 0.5}
    )
    by_id = {r.chunk.chunk_id: r for r in fused}

    # components expose the first-stage scores, not the weighted/normalized contributions:
    # c = 0.1 dense (raw 0.1, normalized 0.0) and 8.0 lexical (raw 8.0, normalized 1.0)
    # b appears only in dense with raw 0.5 (normalized 0.5)
    assert by_id["c"].components == pytest.approx({"dense": 0.1, "lexical": 8.0})
    assert by_id["b"].components == pytest.approx({"dense": 0.5})
    # ... while the fused score is the weighted sum: c = 2.0*0.0 + 0.5*1.0 = 0.5
    assert by_id["c"].score == pytest.approx(0.5)


def test_negative_scores_normalize_to_the_unit_range() -> None:
    dense = [hit("a", 1, -1.0, "dense"), hit("b", 2, -3.0, "dense")]
    lexical = [hit("b", 1, -2.0, "lexical"), hit("c", 2, -4.0, "lexical")]

    fused = weighted_score_fusion({"dense": dense, "lexical": lexical})

    # dense span = -1.0 - (-3.0) = 2.0 -> a = (-1.0 + 3.0)/2.0 = 1.0, b = (-3.0 + 3.0)/2.0 = 0.0
    # lexical span = -2.0 - (-4.0) = 2.0 -> b = (-2.0 + 4.0)/2.0 = 1.0, c = 0.0
    # a = 1.0, b = 0.0 + 1.0 = 1.0, c = 0.0
    assert [r.chunk.chunk_id for r in fused] == ["a", "b", "c"]
    assert [r.score for r in fused] == pytest.approx([1.0, 1.0, 0.0])


def test_top_k_truncates_after_fusion_and_renumbers_ranks() -> None:
    dense = [hit("a", 1, 0.9, "dense"), hit("b", 2, 0.5, "dense"), hit("c", 3, 0.1, "dense")]
    lexical = [hit("c", 1, 8.0, "lexical"), hit("a", 2, 2.0, "lexical")]

    fused = weighted_score_fusion({"dense": dense, "lexical": lexical}, top_k=2)

    # full ordering is a = 1.0, c = 1.0, b = 0.5; top_k=2 keeps the two leaders
    assert [r.chunk.chunk_id for r in fused] == ["a", "c"]
    assert [r.score for r in fused] == pytest.approx([1.0, 1.0])
    assert [r.rank for r in fused] == [1, 2]
