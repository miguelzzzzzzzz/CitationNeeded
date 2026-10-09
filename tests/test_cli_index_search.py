"""End-to-end tests for ``rag-engine index`` and ``rag-engine search``.

Fast tests use the offline hashing embedder; the fastembed path is ``slow``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from rag_engine.cli import _settings_from_args, main

Capture = pytest.CaptureFixture[str]


def build_index(corpus_dir: Path, tmp_path: Path, capsys: Capture, *extra: str) -> Path:
    out = tmp_path / "idx"
    assert main(["index", str(corpus_dir), "--out", str(out), "--embedder", "hashing", *extra]) == 0
    capsys.readouterr()
    return out


def search_json(index: Path, capsys: Capture, *args: str) -> list[dict[str, Any]]:
    assert main(["search", str(index), *args, "--json"]) == 0
    result: list[dict[str, Any]] = json.loads(capsys.readouterr().out)
    return result


def test_index_from_documents(corpus_dir: Path, tmp_path: Path, capsys: Capture) -> None:
    out = tmp_path / "idx"
    assert main(["index", str(corpus_dir), "--out", str(out), "--embedder", "hashing"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "documents": 3,
        "skipped": 0,
        "chunks": 4,
        "embedder": "hashing-512-bigrams",
        "dimension": 512,
        "index_dir": str(out),
    }
    assert (out / "index.json").is_file()
    assert (out / "dense").is_dir() and (out / "lexical").is_dir()


def test_index_from_chunks_jsonl(corpus_dir: Path, tmp_path: Path, capsys: Capture) -> None:
    chunks = tmp_path / "chunks.jsonl"
    assert main(["ingest", str(corpus_dir), "--out", str(chunks)]) == 0
    lines = chunks.read_text(encoding="utf-8").splitlines()
    chunks.write_text("\n\n".join(lines) + "\n\n", encoding="utf-8")  # blank lines ignored
    capsys.readouterr()
    assert main(["index", str(chunks), "--out", str(tmp_path / "i"), "--embedder", "hashing"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["documents"] is None
    assert summary["chunks"] == len(lines) == 4


def test_chunk_overrides_keep_embedding_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: chunking overrides used to rebuild Settings and drop RAG_EMBEDDING_*."""
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "7")
    args = argparse.Namespace(strategy=None, chunk_size=64, chunk_overlap=8)
    settings = _settings_from_args(args)
    assert settings.chunking.chunk_size == 64
    assert settings.embedding.model_name == "BAAI/bge-base-en-v1.5"
    assert settings.embedding.batch_size == 7


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--embedder", "bm25"], "unknown embedder spec"),
        (["--embedder", "hashing", "--chunk-size", "16", "--chunk-overlap", "16"], "invalid"),
    ],
)
def test_index_rejects_bad_options(
    corpus_dir: Path, tmp_path: Path, capsys: Capture, args: list[str], message: str
) -> None:
    assert main(["index", str(corpus_dir), "--out", str(tmp_path / "i"), *args]) == 2
    assert message in capsys.readouterr().err


def test_index_input_errors(tmp_path: Path, capsys: Capture) -> None:
    out = str(tmp_path / "i")
    assert main(["index", str(tmp_path / "missing"), "--out", out]) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"chunk_id": "a"}\nnot json\n', encoding="utf-8")
    assert main(["index", str(bad), "--out", out, "--embedder", "hashing"]) == 2
    assert "line 1" in capsys.readouterr().err  # first line is missing required fields
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    assert main(["index", str(empty), "--out", out, "--embedder", "hashing"]) == 1
    assert "no chunks to index" in capsys.readouterr().err
    assert not (tmp_path / "i" / "index.json").exists()


def test_lexical_search_ranks_phrase_first(
    corpus_dir: Path, tmp_path: Path, capsys: Capture
) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)
    hits = search_json(index, capsys, "reciprocal rank fusion", "--mode", "lexical")
    assert hits[0]["source"] == "retrieval-guide.md"
    assert "Reciprocal rank fusion" in hits[0]["text"]
    assert {h["retriever"] for h in hits} == {"lexical"}
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))

    assert main(["search", str(index), "reciprocal rank fusion", "--mode", "lexical"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("1. ")
    assert "retrieval-guide.md > Retrieval Engineering Guide" in lines[0]
    assert lines[0].endswith(f"[{hits[0]['chunk_id']}]")
    assert lines[1].startswith("    ") and len(lines[1]) <= 4 + 200 + 3
    assert lines[1].endswith("...")  # that chunk is longer than the preview


def test_dense_search_with_hashing(corpus_dir: Path, tmp_path: Path, capsys: Capture) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)
    hits = search_json(index, capsys, "cross-encoder reranking")
    assert hits[0]["source"] == "reranking.html"
    assert hits[0]["retriever"] == "dense"


def test_k_and_filters(corpus_dir: Path, tmp_path: Path, capsys: Capture) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)
    assert len(search_json(index, capsys, "retrieval", "-k", "1")) == 1
    only_notes = search_json(index, capsys, "context boundary", "--filter", "source=notes.txt")
    assert {h["source"] for h in only_notes} == {"notes.txt"}
    two = search_json(
        index,
        capsys,
        "the passage",
        "--filter",
        "source=notes.txt",
        "--filter",
        "source=reranking.html",
    )
    assert {h["source"] for h in two} <= {"notes.txt", "reranking.html"}
    assert "reranking.html" in {h["source"] for h in two}
    for bad in ("source", "page=abc"):
        assert main(["search", str(index), "q", "--filter", bad]) == 2


def test_no_results(corpus_dir: Path, tmp_path: Path, capsys: Capture) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)
    assert main(["search", str(index), "zebra", "--mode", "lexical"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and "no results" in captured.err
    assert search_json(index, capsys, "zebra", "--mode", "lexical") == []
    # a query with no hashed features yields no dense hits instead of an arbitrary ranking
    assert search_json(index, capsys, "?!") == []


def test_search_errors(corpus_dir: Path, tmp_path: Path, capsys: Capture) -> None:
    index = build_index(corpus_dir, tmp_path, capsys)
    assert main(["search", str(index), "q", "-k", "0"]) == 2
    assert main(["search", str(tmp_path / "missing"), "q"]) == 2
    (tmp_path / "plain").mkdir()
    capsys.readouterr()
    assert main(["search", str(tmp_path / "plain"), "q"]) == 2
    assert "not an index directory" in capsys.readouterr().err


@pytest.mark.slow
def test_fastembed_index_and_search(corpus_dir: Path, tmp_path: Path, capsys: Capture) -> None:
    pytest.importorskip("fastembed")
    out = tmp_path / "idx"
    assert main(["index", str(corpus_dir), "--out", str(out)]) == 0
    capsys.readouterr()
    hits = search_json(out, capsys, "cross-encoder reranking")
    assert hits[0]["source"] == "reranking.html"
