"""Tests for :mod:`rag_engine.eval.runner` (fixture-sized, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.eval.runner import (
    EvalConfig,
    build_scifact_documents,
    doc_ranked_list,
    git_sha,
    hardware_note,
    run_evaluation,
    run_structured_eval,
    write_report,
)
from rag_engine.eval.scifact import load_scifact
from rag_engine.models import Chunk
from rag_engine.retrieval.retriever import RetrievedChunk

FIX = Path(__file__).parent / "fixtures" / "eval"
MINI = FIX / "scifact_mini"
REPO_STRUCTURED = Path(__file__).resolve().parents[1] / "evals" / "structured"


def test_hardware_note_and_git_sha() -> None:
    note = hardware_note()
    assert "platform" in note and "python" in note
    assert isinstance(note["cpu_count"], int)
    # We are inside the CitationNeeded checkout when tests run from the repo.
    assert isinstance(git_sha(), str)


def test_doc_ranked_list_dedupes() -> None:
    def hit(doc_id: str, rank: int) -> RetrievedChunk:
        text = "x"
        chunk = Chunk(
            chunk_id=f"{doc_id}#0",
            doc_id=doc_id,
            index=0,
            text=text,
            start_char=0,
            end_char=1,
            token_count=1,
        )
        return RetrievedChunk(chunk=chunk, score=1.0, rank=rank, retriever="dense")

    assert doc_ranked_list([hit("a", 1), hit("b", 2), hit("a", 3)]) == ["a", "b"]


def test_build_scifact_documents_and_eval(tmp_path: Path) -> None:
    split = load_scifact(MINI, split="test", checksum="mini")
    docs = build_scifact_documents(split, corpus_id="mini-eval")
    assert len(docs) == 3
    assert all(doc.format == "txt" for doc in docs)
    beir_to_doc = {str(doc.metadata["beir_doc_id"]): doc.doc_id for doc in docs}
    queries = {qid: q.text for qid, q in split.queries.items()}
    qrels = {
        qid: {beir_to_doc[bid]: g for bid, g in grades.items() if bid in beir_to_doc}
        for qid, grades in split.qrels.items()
    }
    report = run_evaluation(
        documents=docs,
        queries=queries,
        qrels=qrels,
        config=EvalConfig(
            dataset="scifact",
            modes=("dense", "lexical"),
            embedder="hashing",
            top_k=3,
            corpus_id="mini-eval",
        ),
        dataset_checksum="mini",
        dataset_name="scifact/test-mini",
    )
    assert report.n_documents == 3
    assert report.n_queries == 2
    assert report.n_chunks >= 3
    assert set(report.metrics) == {"dense", "lexical"}
    assert "ndcg@10" in report.metrics["lexical"]
    out = write_report(report, tmp_path / "mini.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["dataset_checksum"] == "mini"
    assert "config" in payload and "hardware" in payload


def test_run_structured_eval_writes_report(tmp_path: Path) -> None:
    report = run_structured_eval(
        REPO_STRUCTURED,
        config=EvalConfig(
            dataset="structured",
            modes=("lexical",),
            embedder="hashing",
            chunk_strategy="structure",
            top_k=5,
            corpus_id="structured-eval",
        ),
        results_dir=tmp_path,
    )
    assert report.n_documents == 3
    assert report.n_queries == 8
    assert report.metrics["lexical"]["hit@1"] == pytest.approx(1.0)
    written = list(tmp_path.glob("structured_*.json"))
    assert len(written) == 1


def test_eval_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown modes"):
        EvalConfig(dataset="scifact", modes=("bm25",))  # type: ignore[arg-type]


def test_run_scifact_eval_via_mini_zip(tmp_path: Path) -> None:
    """End-to-end SciFact eval path using the mini fixture over file://."""
    from rag_engine.eval.runner import run_scifact_eval

    url = (FIX / "scifact_mini.zip").resolve().as_uri()
    md5 = (FIX / "scifact_mini.md5").read_text(encoding="utf-8").strip()
    report = run_scifact_eval(
        tmp_path,
        config=EvalConfig(
            dataset="scifact",
            modes=("dense", "lexical", "hybrid"),
            embedder="hashing",
            top_k=3,
            candidates=5,
            corpus_id="mini-eval",
        ),
        results_dir=tmp_path / "out",
        url=url,
        expected_md5=md5,
    )
    assert report.n_documents == 3
    assert report.n_queries == 2
    assert set(report.metrics) == {"dense", "lexical", "hybrid"}
    assert list((tmp_path / "out").glob("scifact_*.json"))


def test_eval_config_rejects_bad_dataset() -> None:
    with pytest.raises(ValueError, match="dataset must be"):
        EvalConfig(dataset="msmarco")


def test_run_scifact_eval_rejects_wrong_dataset(tmp_path: Path) -> None:
    from rag_engine.eval.runner import run_scifact_eval

    with pytest.raises(ValueError, match="expects dataset='scifact'"):
        run_scifact_eval(tmp_path, config=EvalConfig(dataset="structured"))


def test_run_structured_eval_rejects_wrong_dataset(tmp_path: Path) -> None:
    from rag_engine.eval.runner import run_structured_eval

    with pytest.raises(ValueError, match="expects dataset='structured'"):
        run_structured_eval(tmp_path, config=EvalConfig(dataset="scifact"))


def test_run_structured_default_config(tmp_path: Path) -> None:
    report = run_structured_eval(REPO_STRUCTURED, results_dir=tmp_path)
    assert report.config.chunk_strategy == "structure"
    assert "dense" in report.metrics


def test_fingerprint_and_unlabeled_query_skipped(tmp_path: Path) -> None:
    from rag_engine.eval.runner import _fingerprint, run_evaluation

    split = load_scifact(MINI, split="test", checksum="mini")
    docs = build_scifact_documents(split, corpus_id="mini-eval")
    beir_to_doc = {str(doc.metadata["beir_doc_id"]): doc.doc_id for doc in docs}
    queries = {qid: q.text for qid, q in split.queries.items()}
    queries["unlabeled"] = "no judgments for this claim"
    qrels = {
        qid: {beir_to_doc[bid]: g for bid, g in grades.items() if bid in beir_to_doc}
        for qid, grades in split.qrels.items()
    }
    digest = _fingerprint(docs, queries, qrels)
    assert len(digest) == 64
    report = run_evaluation(
        documents=docs,
        queries=queries,
        qrels=qrels,
        config=EvalConfig(dataset="scifact", modes=("lexical",), embedder="hashing", top_k=3),
        dataset_checksum=digest,
        dataset_name="mini",
    )
    assert report.n_queries == 2  # unlabeled excluded from scored count


def test_eval_config_rejects_empty_modes_and_bad_strategy() -> None:
    with pytest.raises(ValueError, match="modes must not be empty"):
        EvalConfig(dataset="scifact", modes=())
    with pytest.raises(ValueError, match="chunk_strategy"):
        EvalConfig(dataset="scifact", chunk_strategy="paragraph")


def test_git_sha_outside_repo(tmp_path: Path) -> None:
    assert git_sha(tmp_path) == ""


def test_percentile_empty_raises() -> None:
    from rag_engine.eval.runner import _percentile

    with pytest.raises(ValueError, match="empty"):
        _percentile([], 0.5)


def test_file_sha256_missing(tmp_path: Path) -> None:
    from rag_engine.eval.runner import _file_sha256

    assert _file_sha256(tmp_path / "nope") == ""


def test_hybrid_rerank_mode_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """hybrid+rerank path without downloading a cross-encoder."""
    from rag_engine.eval import runner as runner_mod
    from rag_engine.retrieval.rerank import RerankingRetriever

    class StubReranker:
        name = "stub-rerank"

        def score(self, query: str, passages: list[str]) -> list[float]:
            # Preserve input order with descending-but-equal-ish scores.
            return [float(len(passages) - i) for i in range(len(passages))]

    monkeypatch.setattr(runner_mod, "CrossEncoderReranker", lambda *a, **k: StubReranker())
    split = load_scifact(MINI, split="test", checksum="mini")
    docs = build_scifact_documents(split, corpus_id="mini-eval")
    beir_to_doc = {str(doc.metadata["beir_doc_id"]): doc.doc_id for doc in docs}
    queries = {qid: q.text for qid, q in split.queries.items()}
    qrels = {
        qid: {beir_to_doc[bid]: g for bid, g in grades.items() if bid in beir_to_doc}
        for qid, grades in split.qrels.items()
    }
    report = run_evaluation(
        documents=docs,
        queries=queries,
        qrels=qrels,
        config=EvalConfig(
            dataset="scifact",
            modes=("hybrid+rerank",),
            embedder="hashing",
            top_k=3,
            candidates=5,
        ),
        dataset_checksum="mini",
        dataset_name="mini",
    )
    assert "hybrid+rerank" in report.metrics
    assert RerankingRetriever  # imported for type presence
