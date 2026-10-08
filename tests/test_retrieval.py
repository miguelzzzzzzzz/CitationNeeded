from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag_engine.chunking import chunk_documents
from rag_engine.config import ChunkingConfig
from rag_engine.ingestion import ingest_path
from rag_engine.models import Chunk
from rag_engine.retrieval import HashingEmbedder, InMemoryVectorStore, matches


def make_chunk(i: int, doc: str = "d1", **metadata: object) -> Chunk:
    return Chunk(
        chunk_id=f"{doc}-{i}",
        doc_id=doc,
        index=i,
        text=f"chunk {i}",
        start_char=0,
        end_char=7,
        token_count=2,
        metadata=dict(metadata),
    )


class TestHashingEmbedder:
    def test_deterministic_and_normalized(self) -> None:
        a = HashingEmbedder(dimension=64).embed_documents(["Hybrid retrieval with BM25", ""])
        b = HashingEmbedder(dimension=64).embed_documents(["Hybrid retrieval with BM25", ""])
        np.testing.assert_array_equal(a, b)
        assert a.dtype == np.float32 and a.shape == (2, 64)
        assert np.linalg.norm(a[0]) == pytest.approx(1.0, abs=1e-6)
        assert not a[1].any()  # empty text -> zero vector, not NaN

    def test_lexical_overlap_increases_similarity(self) -> None:
        emb = HashingEmbedder()
        q = emb.embed_query("reciprocal rank fusion")
        related = emb.embed_query("Reciprocal rank fusion combines ranked lists")
        unrelated = emb.embed_query("PDF pages are parsed one at a time")
        assert float(q @ related) > float(q @ unrelated)

    def test_rejects_tiny_dimension(self) -> None:
        with pytest.raises(ValueError):
            HashingEmbedder(dimension=4)


class TestInMemoryVectorStore:
    def test_search_matches_brute_force_exactly(self) -> None:
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(200, 32)).astype(np.float32)
        store = InMemoryVectorStore(32)
        store.upsert([make_chunk(i) for i in range(200)], vectors)
        for _ in range(5):
            query = rng.normal(size=32).astype(np.float32)
            hits = store.search(query, k=10)
            normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            expected = np.argsort(-(normalized @ (query / np.linalg.norm(query))))[:10]
            assert [h.chunk.index for h in hits] == expected.tolist()
            assert [h.rank for h in hits] == list(range(1, 11))
            scores = [h.score for h in hits]
            assert scores == sorted(scores, reverse=True)

    def test_ties_are_broken_by_insertion_order(self) -> None:
        store = InMemoryVectorStore(2)
        store.upsert([make_chunk(i) for i in range(6)], np.tile([1.0, 0.0], (6, 1)))
        hits = store.search(np.array([1.0, 0.0], dtype=np.float32), k=3)
        assert [h.chunk.index for h in hits] == [0, 1, 2]

    def test_k_larger_than_index_and_empty_index(self) -> None:
        store = InMemoryVectorStore(2)
        q = np.array([1.0, 0.0], dtype=np.float32)
        assert store.search(q, k=5) == []
        store.upsert([make_chunk(0)], np.array([[0.0, 1.0]]))
        assert len(store.search(q, k=5)) == 1

    def test_filters_are_applied_before_ranking(self) -> None:
        store = InMemoryVectorStore(2)
        chunks = [
            make_chunk(0, fmt="pdf", tags=["a"]),
            make_chunk(1, fmt="markdown", tags=["b", "c"]),
            make_chunk(2, fmt="markdown", tags=["c"]),
        ]
        # the best-scoring chunk (0) does not match the filter
        store.upsert(chunks, np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]))
        q = np.array([1.0, 0.0], dtype=np.float32)
        assert [h.chunk.index for h in store.search(q, k=1, filters={"fmt": "markdown"})] == [1]
        assert [h.chunk.index for h in store.search(q, filters={"tags": "c"})] == [1, 2]
        assert [h.chunk.index for h in store.search(q, filters={"tags": ["a", "b"]})] == [0, 1]
        assert store.search(q, filters={"missing": "x"}) == []
        assert [h.chunk.index for h in store.search(q, filters={"doc_id": "d1"})] == [0, 1, 2]

    def test_upsert_replaces_all_chunks_of_a_document(self) -> None:
        store = InMemoryVectorStore(2)
        store.upsert([make_chunk(i, "a") for i in range(3)], np.ones((3, 2)))
        store.upsert([make_chunk(0, "b")], np.ones((1, 2)))
        store.upsert([make_chunk(9, "a")], np.ones((1, 2)))  # doc "a" re-chunked to 1 chunk
        assert sorted(c.chunk_id for c in store.chunks) == ["a-9", "b-0"]
        assert store.delete_document("a") == 1
        assert store.delete_document("a") == 0
        assert len(store) == 1

    @pytest.mark.parametrize(
        ("vectors", "message"),
        [
            (np.ones((2, 3)), "shape"),
            (np.ones((1, 2)), "shape"),
            (np.array([[np.nan, 1.0], [1.0, 1.0]]), "NaN"),
        ],
    )
    def test_invalid_vectors_are_rejected(self, vectors: np.ndarray, message: str) -> None:
        store = InMemoryVectorStore(2)
        with pytest.raises(ValueError, match=message):
            store.upsert([make_chunk(0), make_chunk(1)], vectors)

    def test_invalid_queries_are_rejected(self) -> None:
        store = InMemoryVectorStore(2)
        with pytest.raises(ValueError, match="dimension"):
            store.search(np.ones(3, dtype=np.float32))
        with pytest.raises(ValueError, match="k must be positive"):
            store.search(np.ones(2, dtype=np.float32), k=0)
        with pytest.raises(ValueError, match="duplicate"):
            store.upsert([make_chunk(0), make_chunk(0)], np.ones((2, 2)))

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(1)
        store = InMemoryVectorStore(16)
        store.upsert([make_chunk(i, tags=["x"]) for i in range(20)], rng.normal(size=(20, 16)))
        store.save(tmp_path / "index", embedder_name="test")
        loaded = InMemoryVectorStore.load(tmp_path / "index")
        query = rng.normal(size=16).astype(np.float32)
        assert [(h.chunk, h.score) for h in loaded.search(query, k=5)] == [
            (h.chunk, h.score) for h in store.search(query, k=5)
        ]

    def test_load_detects_inconsistent_files(self, tmp_path: Path) -> None:
        store = InMemoryVectorStore(4)
        store.upsert([make_chunk(0), make_chunk(1)], np.ones((2, 4)))
        store.save(tmp_path)
        lines = (tmp_path / "chunks.jsonl").read_text().splitlines()
        (tmp_path / "chunks.jsonl").write_text(lines[0] + "\n")
        with pytest.raises(ValueError, match="inconsistent"):
            InMemoryVectorStore.load(tmp_path)


def test_matches_heading_path_and_page() -> None:
    chunk = make_chunk(0).model_copy(update={"heading_path": ("Guide", "Setup"), "page": 3})
    assert matches(chunk, {"heading_path": "Setup", "page": [2, 3]})
    assert not matches(chunk, {"page": 1})


def test_end_to_end_ingest_chunk_embed_search(corpus_dir: Path) -> None:
    documents = ingest_path(corpus_dir).documents
    chunks = chunk_documents(
        documents, ChunkingConfig(chunk_size=24, chunk_overlap=0, min_chunk_tokens=0)
    )
    embedder = HashingEmbedder()
    store = InMemoryVectorStore(embedder.dimension)
    store.upsert(chunks, embedder.embed_documents([c.embedding_text for c in chunks]))

    top = store.search(embedder.embed_query("how does reciprocal rank fusion work"), k=1)[0]
    assert top.chunk.heading_path[-1] == "Hybrid search"
    cited = next(d for d in documents if d.doc_id == top.chunk.doc_id)
    assert cited.text[top.chunk.start_char : top.chunk.end_char] == top.chunk.text

    filtered = store.search(
        embedder.embed_query("latency of rerankers"), k=3, filters={"format": "html"}
    )
    assert filtered and all(h.chunk.metadata["format"] == "html" for h in filtered)
