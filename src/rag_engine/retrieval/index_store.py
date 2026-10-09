"""Index directory format: build embedders from specs and persist a hybrid index.

An "index directory" bundles the dense vector store (``dense/``), the BM25
index (``lexical/``), and a small ``index.json`` manifest. The manifest records
the embedder by *spec* rather than by object, so an index reloads in a fresh
process: :func:`embedder_from_spec` inverts ``Embedder.name`` for every
embedder the library can construct, and reloading therefore reproduces the
vectors that were indexed. No model weights are loaded at load time; fastembed
resolves its dimension from its registry and only downloads on first embed.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_engine.config import EmbeddingConfig
from rag_engine.retrieval.bm25 import BM25Index
from rag_engine.retrieval.embedders import Embedder, HashingEmbedder
from rag_engine.retrieval.fastembed_embedder import FastEmbedEmbedder
from rag_engine.retrieval.retriever import DenseRetriever, LexicalRetriever
from rag_engine.retrieval.vector_store import InMemoryVectorStore

__all__ = ["INDEX_FORMAT_VERSION", "embedder_from_spec", "load_index", "save_index"]

INDEX_FORMAT_VERSION = 1

_DENSE_DIR = "dense"
_LEXICAL_DIR = "lexical"
_MANIFEST = "index.json"


def embedder_from_spec(spec: str, config: EmbeddingConfig | None = None) -> Embedder:
    """Rebuild an embedder from its ``name`` plus optional model settings.

    ``"hashing"`` is a convenience alias for the library default and is the one
    spec that does not round-trip (the default produces ``hashing-512-bigrams``).
    Every other spec mirrors ``Embedder.name`` exactly, so a persisted index
    reloads to an embedder that yields identical vectors.
    """
    if spec.startswith("fastembed:"):
        base = config or EmbeddingConfig()
        model_name = spec.partition(":")[2]
        return FastEmbedEmbedder(base.model_copy(update={"model_name": model_name}))
    if spec == "fastembed":
        return FastEmbedEmbedder(config or EmbeddingConfig())
    if spec == "hashing":
        return HashingEmbedder()
    parts = spec.split("-")
    if parts[0] == "hashing" and len(parts) in (2, 3) and parts[1].isdigit():
        if len(parts) == 2:
            return HashingEmbedder(dimension=int(parts[1]), use_bigrams=False)
        if parts[2] == "bigrams":
            return HashingEmbedder(dimension=int(parts[1]), use_bigrams=True)
    raise ValueError(f"unknown embedder spec {spec!r}")


def save_index(directory: str | Path, dense: DenseRetriever, lexical: LexicalRetriever) -> None:
    """Write a self-describing index directory: ``dense/``, ``lexical/``, manifest.

    The two retrievers must cover the same chunk set, since the manifest stores
    a single chunk count and hybrid fusion assumes aligned key spaces; the check
    runs before anything is written so a failed save cannot leave a directory
    that looks loadable.
    """
    dense_count = len(dense.store)
    lexical_count = len(lexical.bm25)
    if dense_count != lexical_count:
        raise ValueError(
            f"dense index holds {dense_count} chunks but lexical index holds "
            f"{lexical_count}; both must be built from the same chunks"
        )
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    dense.store.save(path / _DENSE_DIR, embedder_name=dense.embedder.name)
    lexical.bm25.save(path / _LEXICAL_DIR)
    manifest: dict[str, object] = {
        "format_version": INDEX_FORMAT_VERSION,
        "embedder": dense.embedder.name,
        "dimension": dense.store.dimension,
        "chunks": dense_count,
        "bm25": {"k1": lexical.bm25.k1, "b": lexical.bm25.b},
    }
    (path / _MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_index(
    directory: str | Path, config: EmbeddingConfig | None = None
) -> tuple[DenseRetriever, LexicalRetriever]:
    """Load a saved index directory, returning its dense and lexical retrievers.

    ``config`` only supplies settings the spec does not encode (cache dir,
    batch size, threads); the stored spec wins for the hashing dimension and the
    fastembed model name so the reloaded embedder matches the saved vectors.
    """
    path = Path(directory)
    manifest_path = path / _MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"not an index directory: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != INDEX_FORMAT_VERSION:
        raise ValueError(f"unsupported index format {manifest.get('format_version')!r}")
    store = InMemoryVectorStore.load(path / _DENSE_DIR)
    bm25 = BM25Index.load(path / _LEXICAL_DIR)
    chunks = int(manifest["chunks"])
    if len(store) != chunks or len(bm25) != chunks:
        raise ValueError(
            f"index is inconsistent with manifest: dense={len(store)}, "
            f"lexical={len(bm25)}, expected {chunks}"
        )
    dimension = int(manifest["dimension"])
    if store.dimension != dimension:
        raise ValueError(
            f"store dimension {store.dimension} does not match manifest dimension {dimension}"
        )
    embedder = embedder_from_spec(manifest["embedder"], config)
    return DenseRetriever(embedder, store), LexicalRetriever(bm25)
