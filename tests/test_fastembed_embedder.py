"""Tests for the fastembed adapter.

Fast tests drive the adapter with a small in-test model that records the
batches it receives; that model stands in only for ONNX inference, which is
the external boundary. Tests marked ``slow`` download and run the real
BAAI/bge-small-en-v1.5 model (~67 MB) and are excluded from default CI.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from rag_engine.config import EmbeddingConfig, Settings
from rag_engine.retrieval import FastEmbedEmbedder, InMemoryVectorStore
from rag_engine.retrieval.embedders import HashingEmbedder


class RecordingModel:
    """Deterministic stand-in for fastembed's TextEmbedding (unnormalized output)."""

    def __init__(self, dim: int = 8, bad: str | None = None) -> None:
        self.dim = dim
        self.bad = bad
        self.doc_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []
        self._hash = HashingEmbedder(dimension=dim)

    def _vectors(self, texts: list[str]) -> Iterable[Any]:
        for text in texts:
            vector = self._hash.embed_query(text) * 3.0  # deliberately not unit norm
            if self.bad == "nan":
                vector = vector.copy()
                vector[0] = np.nan
            yield vector.astype(np.float64)
        if self.bad == "extra":
            yield np.ones(self.dim)

    def embed(self, documents: Iterable[str], batch_size: int = 256) -> Iterable[Any]:
        docs = list(documents)
        self.doc_calls.append(docs)
        return list(self._vectors(docs))

    def query_embed(self, query: Iterable[str]) -> Iterable[Any]:
        queries = list(query)
        self.query_calls.append(queries)
        return list(self._vectors(queries))


def make(batch_size: int = 3, **kwargs: Any) -> tuple[FastEmbedEmbedder, RecordingModel]:
    model = RecordingModel(**kwargs)
    embedder = FastEmbedEmbedder(EmbeddingConfig(batch_size=batch_size), model=model, dimension=8)
    return embedder, model


def test_documents_are_embedded_in_bounded_batches_in_order() -> None:
    embedder, model = make(batch_size=3)
    texts = [f"document number {i}" for i in range(7)]
    matrix = embedder.embed_documents(texts)
    assert [len(call) for call in model.doc_calls] == [3, 3, 1]
    assert [t for call in model.doc_calls for t in call] == texts
    assert matrix.shape == (7, 8) and matrix.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)
    # row i corresponds to text i
    expected = HashingEmbedder(dimension=8).embed_documents(texts)
    np.testing.assert_allclose(matrix, expected, atol=1e-6)


def test_empty_texts_become_zero_rows_without_model_calls() -> None:
    embedder, model = make(batch_size=2)
    matrix = embedder.embed_documents(["", "alpha beta", "   ", "gamma"])
    assert model.doc_calls == [["alpha beta", "gamma"]]
    assert not matrix[0].any() and not matrix[2].any()
    assert np.linalg.norm(matrix[1]) == pytest.approx(1.0, abs=1e-6)
    assert embedder.embed_documents(["", " "]).shape == (2, 8)
    assert embedder.embed_documents([]).shape == (0, 8)
    assert not embedder.embed_query("  ").any()
    assert model.query_calls == []


def test_query_prefix_is_applied_to_queries_only() -> None:
    model = RecordingModel()
    config = EmbeddingConfig(query_prefix="Represent this sentence for searching: ")
    embedder = FastEmbedEmbedder(config, model=model, dimension=8)
    embedder.embed_documents(["passage text"])
    embedder.embed_query("what is bm25")
    assert model.doc_calls == [["passage text"]]
    assert model.query_calls == [["Represent this sentence for searching: what is bm25"]]


@pytest.mark.parametrize(("bad", "message"), [("nan", "NaN"), ("extra", "returned 3 vectors")])
def test_bad_model_output_is_rejected(bad: str, message: str) -> None:
    embedder, _ = make(batch_size=2, bad=bad)
    with pytest.raises(RuntimeError, match=message):
        embedder.embed_documents(["a b", "c d"])


def test_dimension_mismatch_is_detected() -> None:
    model = RecordingModel(dim=8)
    embedder = FastEmbedEmbedder(EmbeddingConfig(), model=model, dimension=16)
    with pytest.raises(RuntimeError, match="expected 16"):
        embedder.embed_documents(["text"])


def test_dimension_is_learned_from_output_when_unknown() -> None:
    embedder = FastEmbedEmbedder(EmbeddingConfig(), model=RecordingModel(dim=8))
    assert embedder.embed_documents(["hello world"]).shape == (1, 8)
    assert embedder.dimension == 8


def test_embedding_settings_from_env() -> None:
    settings = Settings.from_env(
        {
            "RAG_EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5",
            "RAG_EMBEDDING_BATCH_SIZE": "64",
            "RAG_EMBEDDING_QUERY_PREFIX": "query: ",
        }
    )
    assert settings.embedding.model_name == "BAAI/bge-base-en-v1.5"
    assert settings.embedding.batch_size == 64
    assert settings.embedding.query_prefix == "query: "  # trailing space preserved
    with pytest.raises(ValueError, match="invalid RAG_"):
        Settings.from_env({"RAG_EMBEDDING_BATCH_SIZE": "0"})


# --------------------------------------------------------------------------- real model


@pytest.fixture(scope="module")
def bge() -> FastEmbedEmbedder:
    pytest.importorskip("fastembed")
    return FastEmbedEmbedder(EmbeddingConfig(batch_size=4, threads=2))


@pytest.mark.slow
def test_real_model_dimension_and_normalization(bge: FastEmbedEmbedder) -> None:
    assert bge.dimension == 384
    matrix = bge.embed_documents(["Hybrid retrieval combines BM25 and dense vectors.", ""])
    assert matrix.shape == (2, 384)
    assert np.linalg.norm(matrix[0]) == pytest.approx(1.0, abs=1e-5)
    assert not matrix[1].any()


@pytest.mark.slow
def test_real_model_batching_does_not_change_vectors(bge: FastEmbedEmbedder) -> None:
    texts = [f"Sentence {i} about retrieval evaluation and rerankers." for i in range(9)]
    batched = bge.embed_documents(texts)
    single = FastEmbedEmbedder(EmbeddingConfig(batch_size=1, threads=2)).embed_documents(texts)
    np.testing.assert_allclose(batched, single, atol=1e-4)


@pytest.mark.slow
def test_real_model_ranks_semantic_match_first(bge: FastEmbedEmbedder) -> None:
    from rag_engine.models import Chunk

    passages = [
        "The cat sat on the warm windowsill all afternoon.",
        "Reciprocal rank fusion merges ranked lists from several retrievers.",
        "Quarterly revenue grew because of strong subscription sales.",
    ]
    chunks = [
        Chunk(
            chunk_id=str(i),
            doc_id=str(i),
            index=0,
            text=t,
            start_char=0,
            end_char=len(t),
            token_count=len(t.split()),
        )
        for i, t in enumerate(passages)
    ]
    store = InMemoryVectorStore(bge.dimension)
    store.upsert(chunks, bge.embed_documents(passages))
    # no lexical overlap with the target passage beyond stop words
    hits = store.search(bge.embed_query("how do I combine results from multiple search systems"))
    assert hits[0].chunk.chunk_id == "1"
