"""Tests for reciprocal rank fusion in :mod:`rag_engine.retrieval.fusion`."""

from __future__ import annotations

import pytest

from rag_engine.models import Chunk
from rag_engine.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from rag_engine.retrieval.retriever import RetrievedChunk


def make_chunk(chunk_id: str) -> Chunk:
    """Build a minimal but valid chunk for fusion tests."""
    text = f"body of {chunk_id}"
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
    """Build a retriever hit at 1-based ``rank`` with ``score``."""
    return RetrievedChunk(chunk=make_chunk(chunk_id), score=score, rank=rank, retriever=retriever)


def example_results() -> dict[str, list[RetrievedChunk]]:
    """The documented example: dense ``[a, b, c]`` against lexical ``[c, a, d]``."""
    return {
        "dense": [
            hit("a", 1, 0.9, "dense"),
            hit("b", 2, 0.8, "dense"),
            hit("c", 3, 0.7, "dense"),
        ],
        "lexical": [
            hit("c", 1, 5.5, "lexical"),
            hit("a", 2, 4.5, "lexical"),
            hit("d", 3, 3.5, "lexical"),
        ],
    }


def test_reciprocal_rank_fusion_scores_documented_example() -> None:
    fused = reciprocal_rank_fusion(example_results())

    # a = 1/61 + 1/62 > c = 1/63 + 1/61 > b = 1/62 > d = 1/63
    assert [result.chunk.chunk_id for result in fused] == ["a", "c", "b", "d"]
    assert [result.rank for result in fused] == [1, 2, 3, 4]
    assert [result.retriever for result in fused] == ["hybrid-rrf"] * 4
    assert fused[0].score == pytest.approx(0.0325224748810153)  # 1/61 + 1/62
    assert fused[1].score == pytest.approx(0.0322664584959667)  # 1/63 + 1/61
    assert fused[2].score == pytest.approx(0.0161290322580645)  # 1/62
    assert fused[3].score == pytest.approx(0.0158730158730159)  # 1/63
    # components keep each list's original score, only for lists that saw the chunk
    assert fused[0].components == {"dense": 0.9, "lexical": 4.5}
    assert fused[1].components == {"dense": 0.7, "lexical": 5.5}
    assert fused[2].components == {"dense": 0.8}
    assert fused[3].components == {"lexical": 3.5}


def test_reciprocal_rank_fusion_weights_lexical_list_more_heavily() -> None:
    fused = reciprocal_rank_fusion(example_results(), weights={"lexical": 2.0})

    # c = 1/63 + 2 * 1/61 = 0.0486599 > a = 1/61 + 2 * 1/62 = 0.0486515
    # > d = 2 * 1/63 = 0.0317460 > b = 1/62 = 0.0161290
    assert [result.chunk.chunk_id for result in fused] == ["c", "a", "d", "b"]
    assert fused[0].score == pytest.approx(0.0486599011189175)  # 1/63 + 2 * 1/61
    assert fused[1].score == pytest.approx(0.0486515071390799)  # 1/61 + 2 * 1/62
    assert fused[2].score == pytest.approx(0.0317460317460317)  # 2 * 1/63
    assert fused[3].score == pytest.approx(0.0161290322580645)  # 1/62
    assert [result.rank for result in fused] == [1, 2, 3, 4]


def test_reciprocal_rank_fusion_with_k_zero_uses_plain_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion(example_results(), k=0)

    # a = 1/1 + 1/2 = 1.5 > c = 1/1 + 1/3 = 1.3333 > b = 1/2 > d = 1/3
    assert [result.chunk.chunk_id for result in fused] == ["a", "c", "b", "d"]
    assert fused[0].score == pytest.approx(1.5)  # 1/1 + 1/2
    assert fused[1].score == pytest.approx(1.3333333333333333)  # 1/1 + 1/3
    assert fused[2].score == pytest.approx(0.5)  # 1/2
    assert fused[3].score == pytest.approx(0.3333333333333333)  # 1/3


def test_reciprocal_rank_fusion_truncates_to_top_k_and_renumbers_ranks() -> None:
    fused = reciprocal_rank_fusion(example_results(), top_k=2)

    assert [result.chunk.chunk_id for result in fused] == ["a", "c"]
    assert [result.rank for result in fused] == [1, 2]
    assert fused[0].score == pytest.approx(0.0325224748810153)  # 1/61 + 1/62
    assert fused[1].score == pytest.approx(0.0322664584959667)  # 1/63 + 1/61


def test_reciprocal_rank_fusion_of_empty_input_is_empty() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"dense": [], "lexical": []}) == []
    assert reciprocal_rank_fusion({"dense": [], "lexical": [hit("z", 1, 2.0, "lexical")]}) != []


def test_reciprocal_rank_fusion_single_list_preserves_input_order() -> None:
    dense = [
        hit("a", 1, 0.9, "dense"),
        hit("b", 2, 0.4, "dense"),
        hit("c", 3, 0.1, "dense"),
    ]
    fused = reciprocal_rank_fusion({"dense": dense})

    assert [result.chunk.chunk_id for result in fused] == ["a", "b", "c"]
    assert [result.rank for result in fused] == [1, 2, 3]
    assert fused[0].score == pytest.approx(0.0163934426229508)  # 1/61
    assert fused[1].score == pytest.approx(0.0161290322580645)  # 1/62
    assert fused[2].score == pytest.approx(0.0158730158730159)  # 1/63
    assert fused[0].components == {"dense": 0.9}


def test_reciprocal_rank_fusion_ties_break_on_best_rank_then_first_appearance() -> None:
    # With k=0 every contribution is a plain 1/rank, so all three chunks score 1.0:
    #   a = 1/1            -> best rank 1, seen first (inside "dense")
    #   b = 1/2 + 1/2      -> best rank 2, seen between a and c
    #   c = 1/1            -> best rank 1, seen last (inside "lexical")
    results = {
        "dense": [hit("a", 1, 0.9, "dense"), hit("b", 2, 0.8, "dense")],
        "lexical": [hit("c", 1, 0.7, "lexical"), hit("b", 2, 0.6, "lexical")],
    }
    fused = reciprocal_rank_fusion(results, k=0)

    # c outranks b on best rank despite b being seen first; a and c tie on both
    # score and best rank, so first appearance decides.
    assert [result.chunk.chunk_id for result in fused] == ["a", "c", "b"]
    assert [result.score for result in fused] == pytest.approx([1.0, 1.0, 1.0])


def test_reciprocal_rank_fusion_rejects_duplicate_chunk_within_one_list() -> None:
    duplicates = [hit("a", 1, 0.9, "dense"), hit("a", 2, 0.5, "dense")]

    with pytest.raises(ValueError, match=r"result list 'dense' contains chunk 'a' more than once"):
        reciprocal_rank_fusion({"dense": duplicates})


def test_reciprocal_rank_fusion_rejects_unknown_weight_name() -> None:
    results = {"dense": [hit("a", 1, 0.9, "dense")]}

    with pytest.raises(ValueError, match=r"unknown result lists: \['lexical'\]"):
        reciprocal_rank_fusion(results, weights={"lexical": 2.0})


def test_reciprocal_rank_fusion_rejects_negative_weight() -> None:
    results = {"dense": [hit("a", 1, 0.9, "dense")]}

    with pytest.raises(
        ValueError, match=r"weight for 'dense' must be a finite non-negative number"
    ):
        reciprocal_rank_fusion(results, weights={"dense": -0.5})


def test_reciprocal_rank_fusion_rejects_negative_k() -> None:
    results = {"dense": [hit("a", 1, 0.9, "dense")]}

    with pytest.raises(ValueError, match="k must be non-negative"):
        reciprocal_rank_fusion(results, k=-1)


def test_reciprocal_rank_fusion_rejects_non_positive_top_k() -> None:
    results = {"dense": [hit("a", 1, 0.9, "dense")]}

    with pytest.raises(ValueError, match="top_k must be positive"):
        reciprocal_rank_fusion(results, top_k=0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_fusion_functions_reject_non_finite_weights(bad: float) -> None:
    results = {"dense": [hit("a", 1, 0.9, "dense")]}
    with pytest.raises(ValueError, match="finite non-negative"):
        reciprocal_rank_fusion(results, weights={"dense": bad})
    with pytest.raises(ValueError, match="finite non-negative"):
        weighted_score_fusion(results, weights={"dense": bad})
