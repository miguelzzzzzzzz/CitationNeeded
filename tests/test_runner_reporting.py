from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pydantic
import pytest

from rag_engine.eval.runner import (
    EvalConfig,
    git_dirty,
    run_evaluation,
    run_structured_eval,
)
from rag_engine.models import Document


def _make_config(
    *,
    depth: int = 10,
    max_depth: int | None = None,
) -> EvalConfig:
    return EvalConfig(
        dataset="scifact",
        modes=("lexical",),
        embedder="hashing",
        top_k=10,
        depth=depth,
        max_depth=max_depth,
        chunk_strategy="fixed",
        chunk_size=64,
        chunk_overlap=0,
        corpus_id="t",
    )


def _documents() -> list[Document]:
    return [
        Document.create(
            source="alpha.txt",
            title="Alpha",
            format="txt",
            text="alpha beta gamma delta",
            corpus_id="t",
        ),
        Document.create(
            source="beta.txt",
            title="Beta",
            format="txt",
            text="beta gamma epsilon zeta",
            corpus_id="t",
        ),
    ]


def test_unlabelled_and_non_relevant_queries_are_counted() -> None:
    documents = _documents()
    first_id = documents[0].doc_id
    second_id = documents[1].doc_id

    queries = {
        "q-unlabelled": "alpha beta",
        "q-non-relevant": "gamma delta",
        "q-labelled": "alpha",
    }
    qrels: dict[str, dict[str, int]] = {
        "q-non-relevant": {first_id: 0},
        "q-labelled": {second_id: 1},
    }

    report = run_evaluation(
        documents=documents,
        queries=queries,
        qrels=qrels,
        config=_make_config(),
        dataset_checksum="abc123",
        dataset_name="scifact",
    )

    assert report.n_queries == 1
    assert report.n_queries_unlabelled == 1
    assert report.n_queries_skipped_no_relevant == 1


def test_eval_config_validation() -> None:
    with pytest.raises(pydantic.ValidationError):
        _make_config(depth=0)
    with pytest.raises(pydantic.ValidationError):
        _make_config(depth=10, max_depth=5)


def test_git_dirty_outside_checkout_and_report_field(tmp_path: Path) -> None:
    assert git_dirty(tmp_path) is None

    documents = _documents()
    queries = {"q": "alpha"}
    qrels: dict[str, dict[str, int]] = {"q": {documents[0].doc_id: 1}}

    report = run_evaluation(
        documents=documents,
        queries=queries,
        qrels=qrels,
        config=_make_config(),
        dataset_checksum="abc123",
        dataset_name="scifact",
    )

    assert report.git_dirty is None or isinstance(report.git_dirty, bool)


def test_run_structured_eval_marks_smoke_report(tmp_path: Path) -> None:
    set_root = Path(__file__).resolve().parents[1] / "evals" / "structured"
    if not set_root.is_dir():
        pytest.skip("structured eval set is not available")

    report = run_structured_eval(
        set_root,
        results_dir=tmp_path,
    )

    assert report.smoke_only is True
    assert "smoke test only" in report.notes

    payloads: list[dict[str, Any]] = []
    for path in tmp_path.rglob("*.json"):
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "smoke_only" in data:
            payloads.append(data)

    assert payloads, "structured eval did not write a report JSON"
    payload = payloads[0]
    assert payload["smoke_only"] is True
    assert "retrieval" in payload


def test_retrieval_warmup_call_count(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_engine.eval.runner as runner_module

    original: Any = runner_module.retrieve_documents
    calls: list[int] = []

    def counting_retrieve(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "retrieve_documents", counting_retrieve)

    documents = _documents()
    queries = {"q-one": "alpha", "q-two": "beta"}
    qrels: dict[str, dict[str, int]] = {
        "q-one": {documents[0].doc_id: 1},
        "q-two": {documents[1].doc_id: 1},
    }

    report = run_evaluation(
        documents=documents,
        queries=queries,
        qrels=qrels,
        config=_make_config(),
        dataset_checksum="abc123",
        dataset_name="scifact",
    )

    assert report.n_queries == 2
    assert len(calls) == 3
