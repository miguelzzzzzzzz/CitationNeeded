"""Index directory format: build embedders from specs and persist a hybrid index.

An "index directory" bundles the dense vector store (``dense/``), the BM25 index
(``lexical/``), the normalized source documents (``documents.jsonl``), and a
small ``index.json`` manifest. Chunk offsets point into the *normalized* text in
``documents.jsonl``, not into raw-file bytes, so a citation can only be resolved
against that file (:func:`load_documents`). The manifest records the embedder by
*spec* rather than by object, so an index reloads in a fresh process:
:func:`embedder_from_spec` inverts ``Embedder.name`` for every embedder the
library can construct, and reloading therefore reproduces the vectors that were
indexed. The manifest also carries the vector dimension, so loading needs neither
fastembed's model registry nor a model download; only the first embed does. The
BM25 tokenizer is not serialized: pass the one used at build time back to
:func:`load_index`. Format 2 changed doc ids to ``make_doc_id(corpus_id,
source)``; format-1 indexes are rejected and must be rebuilt.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_engine.config import EmbeddingConfig
from rag_engine.documents import DOCUMENTS_FILE, read_documents, verify_chunks, write_documents
from rag_engine.models import Document
from rag_engine.retrieval.bm25 import BM25Index, Tokenizer, tokenize
from rag_engine.retrieval.embedders import Embedder, HashingEmbedder
from rag_engine.retrieval.fastembed_embedder import FastEmbedEmbedder
from rag_engine.retrieval.retriever import DenseRetriever, LexicalRetriever
from rag_engine.retrieval.vector_store import InMemoryVectorStore

__all__ = [
    "INDEX_FORMAT_VERSION",
    "embedder_from_spec",
    "load_documents",
    "load_index",
    "save_index",
]

INDEX_FORMAT_VERSION = 2

_DENSE_DIR = "dense"
_LEXICAL_DIR = "lexical"
_MANIFEST = "index.json"


def embedder_from_spec(
    spec: str, config: EmbeddingConfig | None = None, *, dimension: int | None = None
) -> Embedder:
    """Rebuild an embedder from its ``name`` plus optional model settings.

    ``"hashing"`` is a convenience alias for the library default and is the one
    spec that does not round-trip (the default produces ``hashing-512-bigrams``).
    Every other spec mirrors ``Embedder.name`` exactly, so a persisted index
    reloads to an embedder that yields identical vectors.

    ``dimension`` lets the loader rebuild fastembed embedders without consulting
    fastembed's model registry or downloading a model (e.g. lexical-only search
    over an index built with a fastembed embedder while fastembed is not
    installed). Hashing specs already fix their dimension, so a conflicting
    ``dimension`` is rejected rather than silently reinterpreting stored vectors.
    """
    if spec.startswith("fastembed:"):
        model_name = spec.partition(":")[2]
        if not model_name:
            raise ValueError("fastembed spec must name a model, e.g. 'fastembed:bge-small-en'")
        base = config or EmbeddingConfig()
        return FastEmbedEmbedder(
            base.model_copy(update={"model_name": model_name}), dimension=dimension
        )
    if spec == "fastembed":
        return FastEmbedEmbedder(config or EmbeddingConfig(), dimension=dimension)
    embedder: Embedder | None = None
    if spec == "hashing":
        embedder = HashingEmbedder()
    else:
        parts = spec.split("-")
        if parts[0] == "hashing" and len(parts) in (2, 3) and parts[1].isdigit():
            if len(parts) == 2:
                embedder = HashingEmbedder(dimension=int(parts[1]), use_bigrams=False)
            elif parts[2] == "bigrams":
                embedder = HashingEmbedder(dimension=int(parts[1]), use_bigrams=True)
    if embedder is None:
        raise ValueError(f"unknown embedder spec {spec!r}")
    if dimension is not None and dimension != embedder.dimension:
        raise ValueError(
            f"spec {spec!r} fixes the hashing dimension at {embedder.dimension}, "
            f"but {dimension} was requested"
        )
    return embedder


def save_index(
    directory: str | Path,
    dense: DenseRetriever,
    lexical: LexicalRetriever,
    documents: Sequence[Document],
) -> None:
    """Write a self-describing index directory: ``dense/``, ``lexical/``, docs, manifest.

    The two retrievers must cover the same chunk ids, since the manifest stores a
    single chunk count and hybrid fusion assumes aligned key spaces, and every
    chunk offset must resolve inside ``documents``; all checks run before anything
    is written so a failed save cannot leave a directory that looks loadable.
    ``documents`` may hold entries no chunk refers to (an empty file yields no
    chunks). Any existing manifest is deleted first and the new manifest is
    written last, so an interrupted re-save never pairs stale metadata with fresh
    sub-indexes.
    """
    embedder_spec = _persistable_spec(dense.embedder.name, dense.store.dimension)
    dense_ids = {chunk.chunk_id for chunk in dense.store.chunks}
    lexical_ids = {chunk.chunk_id for chunk in lexical.bm25.chunks}
    if dense_ids != lexical_ids:
        raise ValueError(
            f"dense index holds {len(dense.store)} chunks but lexical index holds "
            f"{len(lexical.bm25)}, and {len(dense_ids ^ lexical_ids)} chunk ids differ; "
            "both must be built from the same chunks"
        )
    verify_chunks(dense.store.chunks, documents)
    verify_chunks(lexical.bm25.chunks, documents)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / _MANIFEST
    manifest_path.unlink(missing_ok=True)
    dense.store.save(path / _DENSE_DIR, embedder_name=dense.embedder.name)
    lexical.bm25.save(path / _LEXICAL_DIR)
    write_documents(path / DOCUMENTS_FILE, documents)
    manifest: dict[str, object] = {
        "format_version": INDEX_FORMAT_VERSION,
        "embedder": embedder_spec,
        "dimension": dense.store.dimension,
        "chunks": len(dense.store),
        "documents": len(documents),
        "bm25": {"k1": lexical.bm25.k1, "b": lexical.bm25.b},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_index(
    directory: str | Path,
    config: EmbeddingConfig | None = None,
    *,
    tokenizer: Tokenizer = tokenize,
) -> tuple[DenseRetriever, LexicalRetriever]:
    """Load a saved index directory, returning its dense and lexical retrievers.

    ``config`` only supplies settings the spec does not encode (cache dir, batch
    size, threads); the stored spec wins for the hashing dimension and the
    fastembed model name so the reloaded embedder matches the saved vectors. The
    tokenizer is not part of the on-disk format, so a custom one must be passed
    here again to reproduce the terms the BM25 index was built with. Chunk offsets
    are re-checked against ``documents.jsonl`` so a loaded index can always
    resolve its citations.
    """
    path = Path(directory)
    manifest_path = path / _MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"not an index directory: {directory}")
    manifest = _read_manifest(manifest_path)
    format_version = _manifest_int(manifest, "format_version", minimum=0)
    if format_version < INDEX_FORMAT_VERSION:
        raise ValueError(
            f"index format {format_version} is no longer supported "
            f"(current format is {INDEX_FORMAT_VERSION}: corpus-scoped doc ids and "
            "documents.jsonl); rebuild it with `rag-engine index`"
        )
    if format_version != INDEX_FORMAT_VERSION:
        raise ValueError(
            f"index manifest field 'format_version' is {format_version}, "
            f"expected {INDEX_FORMAT_VERSION}"
        )
    embedder_spec = _manifest_str(manifest, "embedder")
    chunks = _manifest_int(manifest, "chunks", minimum=0)
    documents_count = _manifest_int(manifest, "documents", minimum=0)
    dimension = _manifest_int(manifest, "dimension", minimum=1)
    embedder = embedder_from_spec(embedder_spec, config, dimension=dimension)
    store = InMemoryVectorStore.load(path / _DENSE_DIR)
    bm25 = BM25Index.load(path / _LEXICAL_DIR, tokenizer=tokenizer)
    documents = read_documents(path / DOCUMENTS_FILE)
    if len(documents) != documents_count:
        raise ValueError(
            f"index is inconsistent with manifest: documents={len(documents)}, "
            f"expected {documents_count}"
        )
    if len(store) != chunks or len(bm25) != chunks:
        raise ValueError(
            f"index is inconsistent with manifest: dense={len(store)}, "
            f"lexical={len(bm25)}, expected {chunks}"
        )
    if store.dimension != dimension:
        raise ValueError(
            f"store dimension {store.dimension} does not match manifest dimension {dimension}"
        )
    dense_ids = {chunk.chunk_id for chunk in store.chunks}
    lexical_ids = {chunk.chunk_id for chunk in bm25.chunks}
    if dense_ids != lexical_ids:
        raise ValueError(
            f"index is inconsistent with manifest: dense holds {len(dense_ids)} chunk ids, "
            f"lexical holds {len(lexical_ids)}, and {len(dense_ids ^ lexical_ids)} ids differ"
        )
    verify_chunks(store.chunks, documents)
    verify_chunks(bm25.chunks, documents)
    return DenseRetriever(embedder, store), LexicalRetriever(bm25)


def load_documents(directory: str | Path) -> list[Document]:
    """Return the normalized documents an index's chunk offsets point into.

    Citation offsets are relative to the normalized text, so resolving one needs
    the stored text rather than the original file on disk.
    """
    return read_documents(Path(directory) / DOCUMENTS_FILE)


def _persistable_spec(name: str, dimension: int) -> str:
    """Return ``name`` when :func:`embedder_from_spec` can rebuild it exactly.

    The manifest stores the spec, not the embedder object, so a name the loader
    cannot reconstruct (a custom embedder, or one whose spec disagrees with the
    stored dimension) would otherwise yield a directory that cannot be loaded.
    """
    message = f"embedder {name!r} cannot be persisted because its name is not a known spec"
    try:
        rebuilt = embedder_from_spec(name, dimension=dimension)
    except ValueError as exc:
        raise ValueError(message) from exc
    if rebuilt.name != name:
        raise ValueError(message)
    return name


def _read_manifest(path: Path) -> dict[str, Any]:
    """Parse the manifest, mapping every malformed-input failure to ``ValueError``."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"index manifest {path.name} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"index manifest {path.name} must be a JSON object")
    return raw


def _manifest_str(manifest: dict[str, Any], field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"index manifest field {field!r} must be a non-empty string")
    return value


def _manifest_int(manifest: dict[str, Any], field: str, *, minimum: int) -> int:
    """``bool`` subclasses ``int``, so it is excluded explicitly."""
    value = manifest.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"index manifest field {field!r} must be an integer >= {minimum}")
    return value
