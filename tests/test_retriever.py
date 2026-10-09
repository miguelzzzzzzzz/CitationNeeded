"""Tests for :mod:`rag_engine.retrieval.retriever`."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from rag_engine.config import EmbeddingConfig
from rag_engine.models import Chunk
from rag_engine.retrieval.bm25 import BM25Index, tokenize
from rag_engine.retrieval.embedders import Embedder, HashingEmbedder, Matrix, Vector
from rag_engine.retrieval.fastembed_embedder import FastEmbedEmbedder
from rag_engine.retrieval.retriever import (
    DenseRetriever,
    LexicalRetriever,
    RetrievedChunk,
    Retriever,
)
from rag_engine.retrieval.vector_store import InMemoryVectorStore

CAT_TEXT = "The cat sat on the warm windowsill all afternoon."
FUSION_TEXT = "Reciprocal rank fusion merges ranked lists from several retrievers."
REVENUE_TEXT = "Quarterly revenue grew because of strong subscription sales."


class CountingEmbedder:
    """Embedder that delegates to a real embedder and counts calls to it."""

    def __init__(self, inner: Embedder | None = None) -> None:
        self._inner: Embedder = HashingEmbedder() if inner is None else inner
        self.document_calls = 0
        self.query_calls = 0

    @property
    def name(self) -> str:
        return f"counting:{self._inner.name}"

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        self.document_calls += 1
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> Vector:
        self.query_calls += 1
        return self._inner.embed_query(text)


def make_chunk(doc: str, i: int, text: str, **metadata: object) -> Chunk:
    """Build a chunk whose metadata always carries a ``source`` entry."""
    return Chunk(
        chunk_id=f"{doc}-{i}",
        doc_id=doc,
        index=i,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
        metadata={"source": f"{doc}.md", **metadata},
    )


def _sample_result() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="doc-7",
        doc_id="doc",
        index=7,
        text="alpha beta gamma",
        start_char=10,
        end_char=26,
        token_count=3,
        heading_path=("Guide", "Install"),
        page=4,
        metadata={"source": "doc.md", "topic": "install"},
    )
    return RetrievedChunk(chunk=chunk, score=1.25, rank=3, retriever="dense")


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        make_chunk("alpha", 0, CAT_TEXT, topic="animals"),
        make_chunk("beta", 0, FUSION_TEXT, topic="search"),
        make_chunk("gamma", 0, REVENUE_TEXT, topic="finance"),
    ]


# --------------------------------------------------------------------- dense


def test_dense_index_returns_count_and_populates_store(chunks: list[Chunk]) -> None:
    retriever = DenseRetriever(HashingEmbedder())
    assert retriever.name == "dense"
    assert len(retriever.store) == 0
    assert retriever.index(chunks) == 3
    assert len(retriever.store) == 3
    assert retriever.store.chunks == tuple(chunks)


def test_dense_index_empty_batch_skips_embedder() -> None:
    embedder = CountingEmbedder()
    retriever = DenseRetriever(embedder)
    assert retriever.index([]) == 0
    assert len(retriever.store) == 0
    assert embedder.document_calls == 0


def test_dense_creates_store_with_embedder_dimension() -> None:
    retriever = DenseRetriever(HashingEmbedder(dimension=32))
    assert retriever.store.dimension == 32
    assert retriever.embedder.dimension == 32


def test_dense_uses_injected_store(chunks: list[Chunk]) -> None:
    store = InMemoryVectorStore(512)
    embedder = HashingEmbedder(dimension=512)
    retriever = DenseRetriever(embedder, store)
    assert retriever.store is store
    assert retriever.embedder is embedder
    assert retriever.index(chunks) == 3
    assert len(store) == 3


def test_dense_store_dimension_mismatch_raises() -> None:
    store = InMemoryVectorStore(64)
    with pytest.raises(ValueError, match="does not match embedder dimension"):
        DenseRetriever(HashingEmbedder(dimension=128), store)


def test_dense_retrieve_ranks_matching_chunk_first(chunks: list[Chunk]) -> None:
    retriever = DenseRetriever(HashingEmbedder())
    retriever.index(chunks)
    results = retriever.retrieve("warm cat on the windowsill", k=3)
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].chunk.chunk_id == "alpha-0"
    assert results[0].chunk.text == CAT_TEXT
    assert results[0].score > 0.0
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert {r.retriever for r in results} == {"dense"}
    assert len(retriever.retrieve("cat", k=10)) == 3


def test_dense_retrieve_filters_restrict_results(chunks: list[Chunk]) -> None:
    retriever = DenseRetriever(HashingEmbedder())
    retriever.index(chunks)
    finance = retriever.retrieve("warm cat windowsill", k=10, filters={"topic": "finance"})
    assert [r.chunk.chunk_id for r in finance] == ["gamma-0"]
    assert finance[0].rank == 1
    by_doc = retriever.retrieve("ranked lists", k=10, filters={"doc_id": "beta"})
    assert [r.chunk.chunk_id for r in by_doc] == ["beta-0"]
    any_of = retriever.retrieve("warm cat", k=10, filters={"doc_id": ["alpha", "gamma"]})
    assert {r.chunk.chunk_id for r in any_of} == {"alpha-0", "gamma-0"}
    assert retriever.retrieve("warm cat", k=10, filters={"topic": "missing"}) == []


def test_dense_blank_query_returns_empty_without_embedding(chunks: list[Chunk]) -> None:
    embedder = CountingEmbedder(HashingEmbedder())
    retriever = DenseRetriever(embedder)
    assert retriever.index(chunks) == 3
    assert embedder.document_calls == 1
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   \n\t ") == []
    assert embedder.query_calls == 0


def test_dense_validates_k_before_blank_query_short_circuit() -> None:
    retriever = DenseRetriever(HashingEmbedder())
    with pytest.raises(ValueError, match="k must be positive"):
        retriever.retrieve("   ", k=0)


@pytest.mark.parametrize("k", [0, -1])
def test_dense_rejects_non_positive_k(chunks: list[Chunk], k: int) -> None:
    retriever = DenseRetriever(HashingEmbedder())
    retriever.index(chunks)
    with pytest.raises(ValueError, match="k must be positive"):
        retriever.retrieve("cat", k=k)


# ------------------------------------------------------------------- lexical


def test_lexical_index_returns_count_and_populates_index(chunks: list[Chunk]) -> None:
    retriever = LexicalRetriever()
    assert retriever.name == "lexical"
    assert len(retriever.bm25) == 0
    assert retriever.index(chunks) == 3
    assert len(retriever.bm25) == 3
    assert retriever.bm25.chunks == tuple(chunks)


def test_lexical_index_empty_batch() -> None:
    retriever = LexicalRetriever()
    assert retriever.index([]) == 0
    assert len(retriever.bm25) == 0


def test_lexical_uses_injected_index(chunks: list[Chunk]) -> None:
    index = BM25Index()
    retriever = LexicalRetriever(index)
    assert retriever.bm25 is index
    assert retriever.index(chunks[:2]) == 2
    assert len(index) == 2
    assert retriever.retrieve("cat", k=5)[0].chunk.chunk_id == "alpha-0"


def test_lexical_retrieve_ranks_matching_chunk_first(chunks: list[Chunk]) -> None:
    retriever = LexicalRetriever()
    retriever.index(chunks)
    results = retriever.retrieve("reciprocal rank fusion merges ranked lists", k=3)
    assert results
    assert results[0].chunk.chunk_id == "beta-0"
    assert results[0].chunk.text == FUSION_TEXT
    assert results[0].score > 0.0
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert {r.retriever for r in results} == {"lexical"}
    assert len(retriever.retrieve("cat", k=100)) == 1


def test_lexical_retrieve_only_returns_chunks_sharing_a_query_term(
    chunks: list[Chunk],
) -> None:
    retriever = LexicalRetriever()
    retriever.index(chunks)
    query = "cat subscription"
    query_terms = set(tokenize(query))
    results = retriever.retrieve(query, k=10)
    assert {r.chunk.chunk_id for r in results} == {"alpha-0", "gamma-0"}
    for result in results:
        assert query_terms & set(tokenize(result.chunk.text))


def test_lexical_retrieve_filters_restrict_results(chunks: list[Chunk]) -> None:
    retriever = LexicalRetriever()
    retriever.index(chunks)
    query = "cat subscription fusion"
    only_finance = retriever.retrieve(query, k=10, filters={"topic": "finance"})
    assert [r.chunk.chunk_id for r in only_finance] == ["gamma-0"]
    assert only_finance[0].rank == 1
    by_source = retriever.retrieve(query, k=10, filters={"source": "beta.md"})
    assert [r.chunk.chunk_id for r in by_source] == ["beta-0"]
    assert retriever.retrieve(query, k=10, filters={"topic": "missing"}) == []


def test_lexical_blank_query_returns_empty(chunks: list[Chunk]) -> None:
    retriever = LexicalRetriever()
    retriever.index(chunks)
    assert retriever.retrieve("") == []
    assert retriever.retrieve(" \n\t ") == []


@pytest.mark.parametrize("k", [0, -5])
def test_lexical_rejects_non_positive_k(chunks: list[Chunk], k: int) -> None:
    retriever = LexicalRetriever()
    retriever.index(chunks)
    with pytest.raises(ValueError, match="k must be positive"):
        retriever.retrieve("cat", k=k)


# ---------------------------------------------------------------- result shape


def test_provenance_exact_fields() -> None:
    result = _sample_result()
    assert result.provenance() == {
        "doc_id": "doc",
        "chunk_id": "doc-7",
        "source": "doc.md",
        "content_hash": None,
        "heading_path": ["Guide", "Install"],
        "page": 4,
        "page_end": 4,
        "start_char": 10,
        "end_char": 26,
    }


def test_provenance_defaults_when_metadata_is_empty() -> None:
    chunk = Chunk(
        chunk_id="plain-0",
        doc_id="plain",
        index=0,
        text="hello",
        start_char=0,
        end_char=5,
        token_count=1,
    )
    result = RetrievedChunk(chunk=chunk, score=0.0, rank=1, retriever="lexical")
    provenance = result.provenance()
    assert provenance["source"] is None
    assert provenance["heading_path"] == []
    assert provenance["page"] is None


def test_to_dict_exact_keys_and_values() -> None:
    result = _sample_result()
    expected: dict[str, Any] = {
        "rank": 3,
        "score": 1.25,
        "retriever": "dense",
        "doc_id": "doc",
        "chunk_id": "doc-7",
        "source": "doc.md",
        "content_hash": None,
        "heading_path": ["Guide", "Install"],
        "page": 4,
        "page_end": 4,
        "start_char": 10,
        "end_char": 26,
        "text": "alpha beta gamma",
    }
    payload = result.to_dict()
    assert payload == expected
    assert list(payload) == list(expected)


def test_to_dict_without_text_and_json_round_trip() -> None:
    result = _sample_result()
    without_text = result.to_dict(include_text=False)
    assert "text" not in without_text
    assert without_text == {key: value for key, value in result.to_dict().items() if key != "text"}
    dumped = json.dumps(result.to_dict())
    assert json.loads(dumped) == result.to_dict()


# ------------------------------------------------------------------- protocol


def test_retrievers_satisfy_retriever_protocol(chunks: list[Chunk]) -> None:
    dense_impl = DenseRetriever(HashingEmbedder())
    lexical_impl = LexicalRetriever()
    assert dense_impl.index(chunks) == 3
    assert lexical_impl.index(chunks) == 3
    dense: Retriever = dense_impl
    lexical: Retriever = lexical_impl
    assert dense.name == "dense"
    assert lexical.name == "lexical"
    assert len(dense.retrieve("warm cat", k=10)) == 3
    assert lexical.retrieve("cat")[0].chunk.chunk_id == "alpha-0"


# ----------------------------------------------------------------- real model


@pytest.mark.slow
def test_dense_fastembed_ranks_rrf_passage_first() -> None:
    pytest.importorskip("fastembed")
    retriever = DenseRetriever(FastEmbedEmbedder(EmbeddingConfig(threads=2)))
    passages = [CAT_TEXT, FUSION_TEXT, REVENUE_TEXT]
    chunks = [make_chunk("passages", i, text) for i, text in enumerate(passages)]
    assert retriever.index(chunks) == 3
    results = retriever.retrieve("how do I combine results from multiple search systems", k=3)
    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].chunk.text == FUSION_TEXT


def test_dense_query_with_zero_vector_returns_no_hits(chunks: list[Chunk]) -> None:
    """Punctuation-only queries hash to no features; that must not rank every chunk at 0."""
    retriever = DenseRetriever(HashingEmbedder(dimension=64))
    retriever.index(chunks)
    assert retriever.retrieve("?!") == []
