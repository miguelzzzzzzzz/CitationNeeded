"""Hand-computed unit tests for :mod:`rag_engine.eval.metrics`."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rag_engine.eval.metrics import (
    METRIC_NAMES,
    aggregate_metrics,
    dedupe_preserve_order,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    relevant_ids,
    score_query,
)

# ---------------------------------------------------------------------------
# Scenario A: binary qrels
#   ranked = ["d1", "d2", "d3", "d4", "d5"]
#   qrels  = {"d1": 0, "d2": 1, "d3": 1, "d5": 1, "d9": 1}
#   relevant R = {d2, d3, d5, d9}  (grades > 0), |R| = 4
# ---------------------------------------------------------------------------
RANKED_A: list[str] = ["d1", "d2", "d3", "d4", "d5"]
QRELS_A: dict[str, int] = {"d1": 0, "d2": 1, "d3": 1, "d5": 1, "d9": 1}

MetricFn = Callable[..., float]


def test_relevant_ids_excludes_zero_grades() -> None:
    # d1 has grade 0 -> not relevant; d9 is relevant although it is not ranked.
    assert relevant_ids(QRELS_A) == {"d2", "d3", "d5", "d9"}


def test_relevant_ids_empty_and_all_zero() -> None:
    assert relevant_ids({}) == set()
    assert relevant_ids({"a": 0, "b": 0}) == set()


def test_dedupe_preserve_order_scenario_c() -> None:
    # ["x", "y", "x", "z"] -> keep first occurrence of x, drop the second.
    assert dedupe_preserve_order(["x", "y", "x", "z"]) == ["x", "y", "z"]


def test_dedupe_preserve_order_empty_and_unique() -> None:
    assert dedupe_preserve_order([]) == []
    assert dedupe_preserve_order(["a", "b", "c"]) == ["a", "b", "c"]


def test_dedupe_preserve_order_does_not_mutate_input() -> None:
    ranked = ["x", "y", "x"]
    dedupe_preserve_order(ranked)
    assert ranked == ["x", "y", "x"]


# ---------------------------------------------------------------------------
# recall / hit / mrr / nDCG on scenario A
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.25), (5, 0.75)])
def test_recall_at_k_scenario_a(k: int, expected: float) -> None:
    # |R| = 4.
    # recall@1 = |{d1} ∩ R| / 4 = 0 / 4 = 0.0
    # recall@2 = |{d1, d2} ∩ R| / 4 = 1 / 4 = 0.25
    # recall@5 = |{d1..d5} ∩ R| / 4 = |{d2, d3, d5}| / 4 = 3 / 4 = 0.75
    assert recall_at_k(RANKED_A, QRELS_A, k) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1.0), (5, 1.0)])
def test_hit_at_k_scenario_a(k: int, expected: float) -> None:
    # hit@1: top-1 = {d1}, no relevant doc -> 0.0
    # hit@2: top-2 = {d1, d2} contains relevant d2 -> 1.0
    # hit@5: top-5 contains relevant d2, d3, d5 -> 1.0
    assert hit_at_k(RANKED_A, QRELS_A, k) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.5), (5, 0.5)])
def test_mrr_at_k_scenario_a(k: int, expected: float) -> None:
    # First relevant document is d2 at rank 2.
    # mrr@1 = 0.0 (no relevant doc inside top-1)
    # mrr@2 = 1 / 2 = 0.5
    # mrr@5 = 1 / 2 = 0.5 (d2 is still the first relevant doc)
    assert mrr_at_k(RANKED_A, QRELS_A, k) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("k", "expected"),
    [(1, 0.0), (2, 0.386852807235), (5, 0.592512031965)],
)
def test_ndcg_at_k_scenario_a(k: int, expected: float) -> None:
    # Grades at ranking positions 1..5 are 0, 1, 1, 0, 1.
    # Gains = 2^rel - 1 = 0, 1, 1, 0, 1.
    # DCG@5 = 0/log2(2) + 1/log2(3) + 1/log2(4) + 0/log2(5) + 1/log2(6)
    #        = 0 + 0.630929753571 + 0.5 + 0 + 0.386852807234
    #        = 1.517782560805
    # IDCG@5 = ideal grades 1, 1, 1, 1 (4 relevant docs, zero padding after)
    #         = 1/log2(2) + 1/log2(3) + 1/log2(4) + 1/log2(5)
    #         = 1 + 0.630929753571 + 0.5 + 0.430676558073
    #         = 2.561606311644
    # nDCG@5 = 1.517782560805 / 2.561606311644 = 0.592512031965
    #
    # nDCG@2: DCG = 0/log2(2) + 1/log2(3) = 0.630929753571
    #         IDCG = 1/log2(2) + 1/log2(3) = 1.630929753571
    #         nDCG = 0.630929753571 / 1.630929753571 = 0.386852807235
    # nDCG@1: DCG = 0, IDCG = 1/log2(2) = 1 -> 0 / 1 = 0.0
    assert ndcg_at_k(RANKED_A, QRELS_A, k) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Scenario B: graded qrels for nDCG@3
#   ranked = ["a", "b", "c"], qrels = {"a": 3, "b": 0, "c": 2, "d": 1}
# ---------------------------------------------------------------------------
def test_ndcg_at_3_scenario_b_graded() -> None:
    ranked = ["a", "b", "c"]
    qrels = {"a": 3, "b": 0, "c": 2, "d": 1}
    # gains 2^rel - 1: rank 1 -> 2^3 - 1 = 7, rank 2 -> 2^0 - 1 = 0, rank 3 -> 2^2 - 1 = 3
    # DCG@3 = 7/log2(2) + 0/log2(3) + 3/log2(4) = 7/1 + 0 + 3/2 = 8.5
    # ideal grades sorted desc: 3, 2, 1 (the unranked d has grade 1, still part of the ideal list)
    # IDCG@3 = 7/log2(2) + 3/log2(3) + 1/log2(4)
    #         = 7 + 1.892789260714 + 0.5 = 9.392789260714
    # nDCG@3 = 8.5 / 9.392789260714 = 0.904949505846
    assert ndcg_at_k(ranked, qrels, 3) == pytest.approx(0.904949505846, abs=1e-9)


def test_other_metrics_scenario_b_graded() -> None:
    ranked = ["a", "b", "c"]
    qrels = {"a": 3, "b": 0, "c": 2, "d": 1}
    # relevant R = {a, c, d}, |R| = 3
    # recall@3 = |{a, b, c} ∩ R| / 3 = |{a, c}| / 3 = 2/3 = 0.666666666667
    assert recall_at_k(ranked, qrels, 3) == pytest.approx(2 / 3, abs=1e-9)
    # first relevant doc is a at rank 1 -> 1/1 = 1.0
    assert mrr_at_k(ranked, qrels, 3) == pytest.approx(1.0, abs=1e-9)
    assert hit_at_k(ranked, qrels, 3) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Scenario C: duplicates in the ranking
#   ranked = ["x", "y", "x", "z"], qrels = {"y": 1, "z": 1}
# ---------------------------------------------------------------------------
def test_scenario_c_duplicates_are_dropped() -> None:
    ranked = ["x", "y", "x", "z"]
    qrels = {"y": 1, "z": 1}
    # after dedupe: ["x", "y", "z"], R = {y, z}, |R| = 2
    # recall@2 = |{x, y} ∩ {y, z}| / 2 = 1 / 2 = 0.5
    # recall@3 = |{x, y, z} ∩ {y, z}| / 2 = 2 / 2 = 1.0
    assert recall_at_k(ranked, qrels, 2) == pytest.approx(0.5, abs=1e-9)
    assert recall_at_k(ranked, qrels, 3) == pytest.approx(1.0, abs=1e-9)
    # first relevant doc after dedupe is y at rank 2 -> mrr@3 = 1 / 2 = 0.5
    assert mrr_at_k(ranked, qrels, 3) == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Scenario D: empty ranking / no relevant documents
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("metric", [recall_at_k, hit_at_k, mrr_at_k, ndcg_at_k])
def test_scenario_d_empty_ranking(metric: MetricFn) -> None:
    # |R| = 1 but nothing was retrieved: recall@1 = 0 / 1 = 0.0
    # hit@1 = 0.0 (no hit), mrr@1 = 0.0 (no relevant rank)
    # nDCG@1: DCG = 0, IDCG = 1/log2(2) = 1 -> 0 / 1 = 0.0
    assert metric([], {"a": 1}, 1) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("metric", [recall_at_k, hit_at_k, mrr_at_k, ndcg_at_k])
@pytest.mark.parametrize("qrels", [{}, {"a": 0}])
def test_scenario_d_no_relevant_documents(metric: MetricFn, qrels: dict[str, int]) -> None:
    # |R| = 0: recall@1 = 0 / max(1, 0) = 0.0 (no ZeroDivisionError)
    # hit@1 = 0.0, mrr@1 = 0.0
    # nDCG@1: ideal gain list is empty -> IDCG = 0 -> metric degrades to 0.0
    assert metric(["a"], qrels, 1) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# k validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("metric", [recall_at_k, hit_at_k, mrr_at_k, ndcg_at_k])
@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises_value_error(metric: MetricFn, k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        metric(RANKED_A, QRELS_A, k)


def test_score_query_empty_ks_raises_value_error() -> None:
    with pytest.raises(ValueError):
        score_query(RANKED_A, QRELS_A, ks=())


# ---------------------------------------------------------------------------
# score_query
# ---------------------------------------------------------------------------
def test_score_query_scenario_a_exact_keys_and_values() -> None:
    scores = score_query(RANKED_A, QRELS_A, ks=(1, 2, 5))
    # recall/hit at every k; mrr and ndcg only at max(ks)=5
    assert set(scores) == {
        "recall@1",
        "recall@2",
        "recall@5",
        "hit@1",
        "hit@2",
        "hit@5",
        "mrr@5",
        "ndcg@5",
    }
    # recall: 0/4 = 0.0; 1/4 = 0.25; 3/4 = 0.75
    assert scores["recall@1"] == pytest.approx(0.0, abs=1e-9)
    assert scores["recall@2"] == pytest.approx(0.25, abs=1e-9)
    assert scores["recall@5"] == pytest.approx(0.75, abs=1e-9)
    # hit: no relevant in top-1 -> 0.0; d2 in top-2 and top-5 -> 1.0
    assert scores["hit@1"] == pytest.approx(0.0, abs=1e-9)
    assert scores["hit@2"] == pytest.approx(1.0, abs=1e-9)
    assert scores["hit@5"] == pytest.approx(1.0, abs=1e-9)
    # mrr: first relevant at rank 2 -> 1/2 = 0.5
    assert scores["mrr@5"] == pytest.approx(0.5, abs=1e-9)
    # nDCG@5 = 1.5177825608059992 / 2.5616063116448506 = 0.592512031965
    assert scores["ndcg@5"] == pytest.approx(0.592512031965, abs=1e-9)


def test_score_query_default_ks_scenario_a() -> None:
    scores = score_query(RANKED_A, QRELS_A)
    assert set(scores) == set(METRIC_NAMES)
    # Only 5 documents are ranked, so k=10 sees the same prefix as k=5
    assert scores["recall@1"] == pytest.approx(0.0, abs=1e-9)
    assert scores["recall@5"] == pytest.approx(0.75, abs=1e-9)
    assert scores["recall@10"] == pytest.approx(0.75, abs=1e-9)
    assert scores["hit@10"] == pytest.approx(1.0, abs=1e-9)
    assert scores["mrr@10"] == pytest.approx(0.5, abs=1e-9)
    assert scores["ndcg@10"] == pytest.approx(0.592512031965, abs=1e-9)


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------
def test_aggregate_metrics_averages_two_queries() -> None:
    per_query: dict[str, dict[str, float]] = {
        "q1": {"recall@5": 0.5, "mrr@5": 0.25},
        "q2": {"recall@5": 1.0, "mrr@5": 0.75},
    }
    agg = aggregate_metrics(per_query)
    assert set(agg) == {"recall@5", "mrr@5"}
    # recall@5 = (0.5 + 1.0) / 2 = 0.75
    assert agg["recall@5"] == pytest.approx(0.75, abs=1e-9)
    # mrr@5 = (0.25 + 0.75) / 2 = 0.5
    assert agg["mrr@5"] == pytest.approx(0.5, abs=1e-9)


def test_aggregate_metrics_empty_is_empty_dict() -> None:
    assert aggregate_metrics({}) == {}


def test_aggregate_metrics_from_score_query_results() -> None:
    q_a = score_query(RANKED_A, QRELS_A, ks=(1, 2, 5))
    q_empty = score_query([], {"a": 1}, ks=(1, 2, 5))
    agg = aggregate_metrics({"a": q_a, "empty": q_empty})
    # second query contributes 0.0 everywhere, so each mean is half of scenario A:
    # recall@2 = 0.25 / 2 = 0.125
    assert agg["recall@2"] == pytest.approx(0.125, abs=1e-9)
    # hit@5 = 1.0 / 2 = 0.5
    assert agg["hit@5"] == pytest.approx(0.5, abs=1e-9)
    # nDCG@5 = 0.592512031965 / 2 = 0.296256015982
    assert agg["ndcg@5"] == pytest.approx(0.296256015982, abs=1e-9)
