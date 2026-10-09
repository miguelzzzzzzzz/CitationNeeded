"""Tests for :mod:`rag_engine.retrieval.index_store`.

``HashingEmbedder`` keeps these tests deterministic and offline; the one real
fastembed round trip is marked ``slow`` and skips when the optional dependency
is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.config import EmbeddingConfig
from rag_engine.models import Chunk
from rag_engine.retrieval.embedders import HashingEmbedder
from rag_engine.retrieval.fastembed_embedder import FastEmbedEmbedder
from rag_engine.retrieval.index_store import (
    INDEX_FORMAT_VERSION,
    embedder_from_spec,
    load_index,
    save_index,
)
from rag_engine.retrieval.retriever import DenseRetriever, LexicalRetriever, RetrievedChunk

HASHING_SPEC = "hashing-64"
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


def make_chunk(doc: str, i: int, text: str, **metadata: object) -> Chunk:
    """Build a chunk whose provenance mirrors what ingestion would produce."""
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


def ids_and_scores(hits: list[RetrievedChunk]) -> list[tuple[str, float]]:
    """Reduce retrieval output to the (chunk id, score) pairs under test."""
    return [(hit.chunk.chunk_id, hit.score) for hit in hits]


def build_retrievers(chunks: list[Chunk]) -> tuple[DenseRetriever, LexicalRetriever]:
    """Index ``chunks`` into a fresh dense/lexical pair sharing one chunk set."""
    dense = DenseRetriever(HashingEmbedder(dimension=64, use_bigrams=False))
    lexical = LexicalRetriever()
    assert dense.index(chunks) == len(chunks)
    assert lexical.index(chunks) == len(chunks)
    return dense, lexical


def read_manifest(directory: Path) -> dict[str, object]:
    """Load ``index.json`` with a concrete value type so mypy stays strict."""
    manifest: dict[str, object] = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    return manifest


def edit_manifest(directory: Path, **updates: object) -> None:
    """Overwrite manifest fields in place, simulating a corrupted directory."""
    manifest = read_manifest(directory)
    manifest.update(updates)
    (directory / "index.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        make_chunk("alpha", 0, "retrieval augmented generation with dense vectors"),
        make_chunk("alpha", 1, "bm25 ranks documents by term frequency"),
        make_chunk("beta", 0, "vector search returns nearest neighbours"),
        make_chunk("beta", 1, "signed feature hashing is deterministic"),
    ]


@pytest.fixture
def retrievers(chunks: list[Chunk]) -> tuple[DenseRetriever, LexicalRetriever]:
    return build_retrievers(chunks)


@pytest.fixture
def saved_index(
    tmp_path: Path, retrievers: tuple[DenseRetriever, LexicalRetriever]
) -> tuple[Path, DenseRetriever, LexicalRetriever]:
    dense, lexical = retrievers
    directory = tmp_path / "index"
    save_index(directory, dense, lexical)
    return directory, dense, lexical


# ----------------------------------------------------------------- embedder_from_spec


def test_embedder_from_spec_round_trips_hashing_names() -> None:
    embedders = (
        HashingEmbedder(dimension=64, use_bigrams=False),
        HashingEmbedder(dimension=128),
    )
    for embedder in embedders:
        rebuilt = embedder_from_spec(embedder.name)
        assert isinstance(rebuilt, HashingEmbedder)
        assert rebuilt.name == embedder.name
        assert rebuilt.dimension == embedder.dimension


def test_embedder_from_spec_parses_hashing_specs() -> None:
    unigrams = embedder_from_spec("hashing-64")
    assert isinstance(unigrams, HashingEmbedder)
    assert unigrams.dimension == 64
    assert unigrams.name == "hashing-64"
    query = "dense vectors"
    expected = HashingEmbedder(dimension=64, use_bigrams=False)
    assert unigrams.embed_query(query).tolist() == expected.embed_query(query).tolist()

    bigrams = embedder_from_spec("hashing-128-bigrams")
    assert isinstance(bigrams, HashingEmbedder)
    assert bigrams.dimension == 128
    assert bigrams.name == "hashing-128-bigrams"


def test_embedder_from_spec_hashing_alias_is_the_default_embedder() -> None:
    embedder = embedder_from_spec("hashing")
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.name == "hashing-512-bigrams"
    assert embedder.dimension == 512


def test_embedder_from_spec_bare_fastembed_uses_default_model() -> None:
    embedder = embedder_from_spec("fastembed")
    assert isinstance(embedder, FastEmbedEmbedder)
    assert embedder.name == f"fastembed:{DEFAULT_FASTEMBED_MODEL}"
    assert embedder.config.model_name == DEFAULT_FASTEMBED_MODEL
    assert embedder.config.batch_size == 32


def test_embedder_from_spec_bare_fastembed_keeps_supplied_config() -> None:
    config = EmbeddingConfig(batch_size=4, threads=2)
    embedder = embedder_from_spec("fastembed", config)
    assert isinstance(embedder, FastEmbedEmbedder)
    assert embedder.name == f"fastembed:{DEFAULT_FASTEMBED_MODEL}"
    assert embedder.config.batch_size == 4
    assert embedder.config.threads == 2


def test_embedder_from_spec_fastembed_overrides_only_the_model_name() -> None:
    config = EmbeddingConfig(batch_size=7, cache_dir="/tmp/rag-cache", threads=3)
    embedder = embedder_from_spec("fastembed:BAAI/bge-base-en-v1.5", config)
    assert isinstance(embedder, FastEmbedEmbedder)
    assert embedder.name == "fastembed:BAAI/bge-base-en-v1.5"
    assert embedder.config.model_name == "BAAI/bge-base-en-v1.5"
    assert embedder.config.batch_size == 7
    assert embedder.config.cache_dir == "/tmp/rag-cache"
    assert embedder.config.threads == 3


@pytest.mark.parametrize("spec", ["", "bm25", "hashing-x", "hashing-64-trigrams"])
def test_embedder_from_spec_rejects_unknown_specs(spec: str) -> None:
    with pytest.raises(ValueError, match="unknown embedder spec"):
        embedder_from_spec(spec)


# ----------------------------------------------------------------------- save / load


def test_retrievers_rank_expected_chunks(
    retrievers: tuple[DenseRetriever, LexicalRetriever],
) -> None:
    dense, lexical = retrievers
    dense_hits = dense.retrieve("retrieval", k=3)
    assert dense_hits[0].chunk.chunk_id == "alpha-0"
    assert dense_hits[0].score > 0.0

    lexical_hits = lexical.retrieve("frequency", k=3)
    assert lexical_hits[0].chunk.chunk_id == "alpha-1"
    assert lexical_hits[0].score > 0.0


def test_save_index_writes_expected_manifest(
    chunks: list[Chunk], saved_index: tuple[Path, DenseRetriever, LexicalRetriever]
) -> None:
    directory, _, lexical = saved_index
    assert (directory / "dense").is_dir()
    assert (directory / "lexical").is_dir()

    manifest = read_manifest(directory)
    assert manifest["format_version"] == INDEX_FORMAT_VERSION == 1
    assert manifest["embedder"] == HASHING_SPEC
    assert manifest["dimension"] == 64
    assert manifest["chunks"] == len(chunks) == 4
    assert manifest["bm25"] == {"k1": lexical.bm25.k1, "b": lexical.bm25.b}


def test_save_load_round_trip_preserves_retrieval_results(
    tmp_path: Path, retrievers: tuple[DenseRetriever, LexicalRetriever]
) -> None:
    dense, lexical = retrievers
    directory = tmp_path / "index"
    save_index(directory, dense, lexical)

    loaded_dense, loaded_lexical = load_index(directory, EmbeddingConfig(batch_size=9))

    assert loaded_dense.embedder.name == dense.embedder.name == HASHING_SPEC
    assert loaded_dense.store.dimension == dense.store.dimension == 64
    assert loaded_lexical.bm25.k1 == lexical.bm25.k1
    assert loaded_lexical.bm25.b == lexical.bm25.b

    before_dense = ids_and_scores(dense.retrieve("retrieval", k=3))
    after_dense = ids_and_scores(loaded_dense.retrieve("retrieval", k=3))
    assert after_dense == before_dense
    assert before_dense[0][0] == "alpha-0"
    assert before_dense[0][1] > 0.0

    before_lexical = ids_and_scores(lexical.retrieve("frequency", k=3))
    after_lexical = ids_and_scores(loaded_lexical.retrieve("frequency", k=3))
    assert after_lexical == before_lexical
    assert before_lexical[0][0] == "alpha-1"
    assert before_lexical[0][1] > 0.0


def test_save_index_rejects_mismatched_chunk_counts(tmp_path: Path, chunks: list[Chunk]) -> None:
    dense = DenseRetriever(HashingEmbedder(dimension=64, use_bigrams=False))
    lexical = LexicalRetriever()
    assert dense.index(chunks) == 4
    assert lexical.index(chunks[:2]) == 2

    directory = tmp_path / "index"
    with pytest.raises(ValueError, match="both must be built from the same chunks"):
        save_index(directory, dense, lexical)
    assert not (directory / "index.json").exists()
    assert not (directory / "dense").exists()
    assert not (directory / "lexical").exists()


def test_load_index_rejects_directory_without_manifest(tmp_path: Path) -> None:
    directory = tmp_path / "not-an-index"
    directory.mkdir()
    with pytest.raises(ValueError, match="not an index directory"):
        load_index(directory)


def test_load_index_rejects_unknown_format_version(
    saved_index: tuple[Path, DenseRetriever, LexicalRetriever],
) -> None:
    directory, _, _ = saved_index
    edit_manifest(directory, format_version=INDEX_FORMAT_VERSION + 1)
    with pytest.raises(ValueError, match="unsupported index format"):
        load_index(directory)


def test_load_index_rejects_manifest_chunk_count_mismatch(
    saved_index: tuple[Path, DenseRetriever, LexicalRetriever],
) -> None:
    directory, _, _ = saved_index
    edit_manifest(directory, chunks=99)
    with pytest.raises(ValueError, match="index is inconsistent with manifest"):
        load_index(directory)


@pytest.mark.slow
def test_fastembed_index_round_trip(tmp_path: Path, chunks: list[Chunk]) -> None:
    pytest.importorskip("fastembed")
    config = EmbeddingConfig(model_name=DEFAULT_FASTEMBED_MODEL, batch_size=8)
    dense = DenseRetriever(FastEmbedEmbedder(config))
    lexical = LexicalRetriever()
    assert dense.index(chunks) == len(chunks)
    assert lexical.index(chunks) == len(chunks)

    directory = tmp_path / "fastembed-index"
    save_index(directory, dense, lexical)
    loaded_dense, _ = load_index(directory, config)

    assert loaded_dense.embedder.name == f"fastembed:{DEFAULT_FASTEMBED_MODEL}"
    assert ids_and_scores(loaded_dense.retrieve("dense vector retrieval", k=2)) == ids_and_scores(
        dense.retrieve("dense vector retrieval", k=2)
    )
