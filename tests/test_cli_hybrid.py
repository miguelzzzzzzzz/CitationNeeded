"""Tests for the hybrid and reranking options of ``rag-engine search``."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rag_engine.cli import main
from rag_engine.retrieval.rerank import CrossEncoderReranker

Capture = pytest.CaptureFixture[str]


def build_index(corpus_dir: Path, tmp_path: Path, capsys: Capture, *extra: str) -> Path:
    """Build a hashing-embedder index of the fixture corpus and return its path."""
    out = tmp_path / "idx"
    assert main(["index", str(corpus_dir), "--out", str(out), "--embedder", "hashing", *extra]) == 0
    capsys.readouterr()
    return out


def search_json(index: Path, capsys: Capture, *args: str) -> list[dict[str, Any]]:
    """Run ``search --json`` and return the parsed list of hits."""
    assert main(["search", str(index), *args, "--json"]) == 0
    result: list[dict[str, Any]] = json.loads(capsys.readouterr().out)
    return result


def chunk_of(hit: dict[str, Any]) -> dict[str, Any]:
    """The chunk payload of a hit (nested under ``chunk`` or flattened into the hit)."""
    nested = hit.get("chunk")
    return nested if isinstance(nested, dict) else hit


def chunk_id(hit: dict[str, Any]) -> str:
    return str(chunk_of(hit)["chunk_id"])


def chunk_text(hit: dict[str, Any]) -> str:
    return str(chunk_of(hit)["text"]).lower()


def _overlap_score(self: object, query: str, texts: Sequence[str]) -> list[float]:
    """Stand-in cross-encoder: 1.0 when the text mentions overlap, else 0.0."""
    del self, query
    return [1.0 if "overlap" in text.lower() else 0.0 for text in texts]


def test_hybrid_json_hits(corpus_dir: Path, tmp_path: Path, capsys: Capture) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)
    hits = search_json(index, capsys, "--mode", "hybrid", "--top-k", "4", "retrieval ranking")

    assert [hit["rank"] for hit in hits] == [1, 2, 3, 4]
    assert {hit["retriever"] for hit in hits} == {"hybrid-rrf"}
    ids = [chunk_id(hit) for hit in hits]
    assert len(set(ids)) == len(ids)
    assert all(set(hit["components"]) <= {"dense", "lexical"} for hit in hits)


def test_hybrid_rrf_promotes_the_fusion_chunk(
    corpus_dir: Path, tmp_path: Path, capsys: Capture
) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)
    hits = search_json(index, capsys, "--mode", "hybrid", "reciprocal rank fusion")

    top = hits[0]
    assert "reciprocal rank fusion" in chunk_text(top)
    assert set(top["components"]) == {"dense", "lexical"}


def test_weighted_fusion_retriever_name(corpus_dir: Path, tmp_path: Path, capsys: Capture) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)
    hits = search_json(
        index, capsys, "--mode", "hybrid", "--fusion", "weighted", "retrieval ranking"
    )

    assert hits
    assert {hit["retriever"] for hit in hits} == {"hybrid-weighted"}


def test_zero_lexical_weight_matches_dense_order(
    corpus_dir: Path, tmp_path: Path, capsys: Capture
) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)
    weighted = search_json(
        index,
        capsys,
        "--mode",
        "hybrid",
        "--fusion",
        "weighted",
        "--weight",
        "dense=1",
        "--weight",
        "lexical=0",
        "--top-k",
        "4",
        "retrieval ranking",
    )
    dense = search_json(index, capsys, "--mode", "dense", "--top-k", "4", "retrieval ranking")

    assert len(dense) == 4
    assert [chunk_id(hit) for hit in weighted] == [chunk_id(hit) for hit in dense]


def test_hybrid_human_output_shows_components(
    corpus_dir: Path, tmp_path: Path, capsys: Capture
) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)
    assert main(["search", str(index), "--mode", "hybrid", "reciprocal rank fusion"]) == 0
    lines = capsys.readouterr().out.splitlines()

    assert lines
    assert "(dense=" in lines[0] or "(lexical=" in lines[0]


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(["--fusion", "weighted"], id="fusion-without-hybrid-mode"),
        pytest.param(["--mode", "hybrid", "--rrf-k", "-1"], id="negative-rrf-k"),
        pytest.param(["--mode", "hybrid", "--candidates", "0"], id="zero-candidates"),
        pytest.param(["--mode", "hybrid", "--weight", "foo=1"], id="unknown-weight-name"),
        pytest.param(["--mode", "hybrid", "--weight", "dense=-1"], id="negative-weight"),
        pytest.param(["--mode", "hybrid", "--weight", "dense=abc"], id="non-numeric-weight"),
    ],
)
def test_search_option_errors_exit_2(
    extra: list[str], corpus_dir: Path, tmp_path: Path, capsys: Capture
) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)
    assert main(["search", str(index), "retrieval ranking", *extra]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")


def test_rerank_puts_matching_chunk_first(
    corpus_dir: Path, tmp_path: Path, capsys: Capture, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)
    monkeypatch.setattr(CrossEncoderReranker, "score", _overlap_score)

    lexical = search_json(index, capsys, "--mode", "lexical", "--top-k", "4", "chunk retrieval")
    assert any("overlap" in chunk_text(hit) for hit in lexical)

    hits = search_json(
        index, capsys, "--mode", "lexical", "--rerank", "--top-k", "4", "chunk retrieval"
    )
    assert hits[0]["retriever"] == "lexical+rerank"
    assert "overlap" in chunk_text(hits[0])
    assert {"lexical", "base_rank"} <= set(hits[0]["components"])


def test_rerank_reports_missing_fastembed(
    corpus_dir: Path, tmp_path: Path, capsys: Capture, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)

    def fail_load(self: object) -> None:
        raise ImportError("fastembed is not installed")

    monkeypatch.setattr(CrossEncoderReranker, "_load", fail_load)
    code = main(
        ["search", str(index), "retrieval ranking", "--mode", "dense", "--rerank", "--json"]
    )

    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "fastembed is not installed" in captured.err


@pytest.mark.slow
def test_real_cross_encoder_reranks_hybrid_results(
    corpus_dir: Path, tmp_path: Path, capsys: Capture
) -> None:
    pytest.importorskip("fastembed")
    index = build_index(corpus_dir, tmp_path, capsys)
    hits = search_json(
        index,
        capsys,
        "--mode",
        "hybrid",
        "--rerank",
        "--top-k",
        "4",
        "why is a cross-encoder more precise than a bi-encoder",
    )

    assert hits
    assert "cross-encoder reranking" in chunk_text(hits[0])
