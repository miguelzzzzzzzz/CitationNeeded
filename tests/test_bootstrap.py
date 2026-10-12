from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.eval.bootstrap import BootstrapCI, load_per_query_scores, paired_bootstrap_ci


def test_paired_bootstrap_ci_known_mean_diff() -> None:
    scores_a = [1.0, 2.0, 3.0]
    scores_b = [0.0, 1.0, 2.0]
    ci = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=1000, confidence=0.95, seed=42)
    assert isinstance(ci, BootstrapCI)
    assert ci.n == 3
    assert ci.mean_diff == pytest.approx(1.0)
    assert ci.ci_low > 0.0
    assert ci.ci_high >= ci.ci_low
    assert ci.confidence == 0.95
    assert ci.n_bootstrap == 1000
    assert ci.seed == 42
    assert ci.n_positive + ci.n_negative <= 1000


def test_paired_bootstrap_ci_hand_checkable_above_zero() -> None:
    scores_a = [1.0, 1.0, 1.0]
    scores_b = [0.0, 0.0, 0.0]
    ci = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=1000, seed=0)
    assert ci.mean_diff == pytest.approx(1.0)
    assert ci.ci_low == pytest.approx(1.0)
    assert ci.ci_high == pytest.approx(1.0)
    assert ci.n_positive == 1000
    assert ci.n_negative == 0


def test_paired_bootstrap_ci_equal_scores_contains_zero() -> None:
    scores_a = [0.5, 0.2, 0.9]
    scores_b = [0.5, 0.2, 0.9]
    ci = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=1000, seed=0)
    assert ci.mean_diff == pytest.approx(0.0)
    assert ci.ci_low <= 0.0 <= ci.ci_high


def test_determinism() -> None:
    scores_a = [0.9, 0.8, 0.7, 0.6, 0.5]
    scores_b = [0.1, 0.2, 0.3, 0.4, 0.5]
    ci1 = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=500, seed=123)
    ci2 = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=500, seed=123)
    assert ci1 == ci2


def test_different_seed_honored() -> None:
    scores_a = [0.9, 0.8, 0.7, 0.6, 0.5]
    scores_b = [0.1, 0.2, 0.3, 0.4, 0.5]
    ci1 = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=500, seed=1)
    ci2 = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=500, seed=2)
    assert ci1.mean_diff == pytest.approx(ci2.mean_diff)
    assert ci1.seed == 1
    assert ci2.seed == 2
    assert ci1.ci_low <= ci1.ci_high
    assert ci2.ci_low <= ci2.ci_high


def test_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_ci([1.0, 2.0], [1.0])


def test_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        paired_bootstrap_ci([], [])


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1])
def test_bad_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap_ci([1.0], [0.0], confidence=confidence)


@pytest.mark.parametrize("n_bootstrap", [0, -1])
def test_bad_n_bootstrap(n_bootstrap: int) -> None:
    with pytest.raises(ValueError, match="n_bootstrap"):
        paired_bootstrap_ci([1.0], [0.0], n_bootstrap=n_bootstrap)


def _write_sidecar(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "aggregate_report": "agg.json",
        "query_ids": ["q1", "q2", "q3"],
        "modes": {
            "dense": {
                "q1": {"ndcg@10": 0.5, "mrr@10": 0.4},
                "q2": {"ndcg@10": 0.2, "mrr@10": 0.1},
                "q3": {"ndcg@10": 0.0, "mrr@10": 0.0},
            },
            "hybrid": {
                "q1": {"ndcg@10": 0.55, "mrr@10": 0.45},
                "q2": {"ndcg@10": 0.25, "mrr@10": 0.15},
                "q3": {"ndcg@10": 0.1, "mrr@10": 0.05},
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_per_query_scores_round_trip(tmp_path: Path) -> None:
    sidecar = tmp_path / "scores.per_query.json"
    _write_sidecar(sidecar)
    scores_a, scores_b, query_ids = load_per_query_scores(
        sidecar, mode_a="hybrid", mode_b="dense", metric="ndcg@10"
    )
    assert query_ids == ["q1", "q2", "q3"]
    assert scores_a == pytest.approx([0.55, 0.25, 0.1])
    assert scores_b == pytest.approx([0.5, 0.2, 0.0])


def test_load_per_query_scores_missing_mode(tmp_path: Path) -> None:
    sidecar = tmp_path / "scores.per_query.json"
    _write_sidecar(sidecar)
    with pytest.raises(ValueError, match=r"mode .* not found"):
        load_per_query_scores(sidecar, mode_a="hybrid", mode_b="bm25", metric="ndcg@10")


def test_load_per_query_scores_missing_metric(tmp_path: Path) -> None:
    sidecar = tmp_path / "scores.per_query.json"
    _write_sidecar(sidecar)
    with pytest.raises(ValueError, match=r"metric .* missing"):
        load_per_query_scores(sidecar, mode_a="hybrid", mode_b="dense", metric="recall@10")


def test_load_per_query_scores_mismatched_ids_without_query_ids(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "modes": {
            "dense": {"q1": {"ndcg@10": 0.5}},
            "hybrid": {"q1": {"ndcg@10": 0.55}, "q2": {"ndcg@10": 0.25}},
        },
    }
    sidecar = tmp_path / "bad.per_query.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="query id sets differ"):
        load_per_query_scores(sidecar, mode_a="hybrid", mode_b="dense", metric="ndcg@10")
