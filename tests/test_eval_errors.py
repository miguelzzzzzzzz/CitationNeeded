from __future__ import annotations

import pytest

from rag_engine.eval.runner import (
    EvalConfig,
    _build_reranker,
    _retrieve_hits,
    git_dirty,
)
from rag_engine.retrieval.retriever import RetrievedChunk
from rag_engine.retrieval.vector_store import Filters


class _StubRetriever:
    name: str = "stub"

    def retrieve(
        self,
        query: str,
        k: int = 10,
        filters: Filters | None = None,
    ) -> list[RetrievedChunk]:
        raise ImportError("no embeddings extra")


def _raise_import_error(*args: object, **kwargs: object) -> None:
    raise ImportError("no embeddings extra")


def _raise_os_error(*args: object, **kwargs: object) -> None:
    raise OSError("git not found")


def test_build_reranker_maps_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rag_engine.eval.runner.CrossEncoderReranker",
        _raise_import_error,
    )
    with pytest.raises(RuntimeError, match="embeddings extra"):
        _build_reranker(EvalConfig(dataset="scifact"))


def test_retrieve_hits_error_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = _StubRetriever()
    with pytest.raises(ImportError):
        _retrieve_hits(retriever, "q", 3, reranking=False)

    monkeypatch.setattr(
        "rag_engine.eval.runner.CrossEncoderReranker",
        _raise_import_error,
    )
    with pytest.raises(RuntimeError, match="embeddings extra"):
        _retrieve_hits(retriever, "q", 3, reranking=True)


def test_git_dirty_returns_none_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rag_engine.eval.runner.subprocess.run",
        _raise_os_error,
    )
    assert git_dirty() is None
