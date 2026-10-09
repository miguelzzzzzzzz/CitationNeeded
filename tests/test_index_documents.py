from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.chunking import chunk_documents
from rag_engine.config import ChunkingConfig, IngestionConfig
from rag_engine.documents import DOCUMENTS_FILE
from rag_engine.ingestion import ingest_path
from rag_engine.models import Chunk, Document
from rag_engine.retrieval.embedders import HashingEmbedder
from rag_engine.retrieval.index_store import (
    INDEX_FORMAT_VERSION,
    load_documents,
    load_index,
    save_index,
)
from rag_engine.retrieval.retriever import DenseRetriever, LexicalRetriever

MANIFEST_FILE = "index.json"


def build(tmp_path: Path) -> tuple[list[Document], list[Chunk]]:
    root = tmp_path / "kb"
    (root / "guide").mkdir(parents=True)
    (root / "guide" / "intro.md").write_text(
        "# Intro\n\nThe first paragraph explains retrieval.\r\n\r\n"
        "## Setup\n\nInstall the package and run the index command.\n",
        encoding="utf-8",
    )
    (root / "notes.txt").write_text("Plain notes about BM25 and dense vectors.\n", encoding="utf-8")
    report = ingest_path(root, IngestionConfig(corpus_id="handbook"))
    chunks = chunk_documents(report.documents, ChunkingConfig(chunk_size=16, chunk_overlap=4))
    return report.documents, chunks


def build_retrievers(chunks: list[Chunk]) -> tuple[DenseRetriever, LexicalRetriever]:
    dense = DenseRetriever(HashingEmbedder(dimension=64, use_bigrams=False))
    lexical = LexicalRetriever()
    dense.index(chunks)
    lexical.index(chunks)
    return dense, lexical


def store_index(index_dir: Path, documents: list[Document], chunks: list[Chunk]) -> None:
    dense, lexical = build_retrievers(chunks)
    save_index(index_dir, dense, lexical, documents)


def read_manifest(index_dir: Path) -> dict[str, object]:
    payload = (index_dir / MANIFEST_FILE).read_text(encoding="utf-8")
    data: dict[str, object] = json.loads(payload)
    return data


def write_manifest(index_dir: Path, manifest: dict[str, object]) -> None:
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def kb(tmp_path: Path) -> tuple[list[Document], list[Chunk]]:
    return build(tmp_path)


def test_save_index_writes_documents_and_manifest(
    tmp_path: Path, kb: tuple[list[Document], list[Chunk]]
) -> None:
    documents, chunks = kb
    index_dir = tmp_path / "index"

    store_index(index_dir, documents, chunks)

    assert (index_dir / DOCUMENTS_FILE).is_file()
    manifest = read_manifest(index_dir)
    assert manifest["format_version"] == INDEX_FORMAT_VERSION == 2
    assert manifest["documents"] == len(documents)


def test_load_documents_round_trips(tmp_path: Path, kb: tuple[list[Document], list[Chunk]]) -> None:
    documents, chunks = kb
    index_dir = tmp_path / "index"
    store_index(index_dir, documents, chunks)

    assert load_documents(index_dir) == documents


def test_load_index_rejects_format_version_1(
    tmp_path: Path, kb: tuple[list[Document], list[Chunk]]
) -> None:
    documents, chunks = kb
    index_dir = tmp_path / "index"
    store_index(index_dir, documents, chunks)

    manifest = read_manifest(index_dir)
    manifest["format_version"] = 1
    write_manifest(index_dir, manifest)

    with pytest.raises(ValueError, match="rebuild"):
        load_index(index_dir)


def test_load_index_rejects_document_count_mismatch(
    tmp_path: Path, kb: tuple[list[Document], list[Chunk]]
) -> None:
    documents, chunks = kb
    index_dir = tmp_path / "index"
    store_index(index_dir, documents, chunks)

    manifest = read_manifest(index_dir)
    manifest["documents"] = len(documents) + 1
    write_manifest(index_dir, manifest)

    with pytest.raises(ValueError, match="documents="):
        load_index(index_dir)


def test_load_index_rejects_tampered_documents_jsonl(
    tmp_path: Path, kb: tuple[list[Document], list[Chunk]]
) -> None:
    documents, chunks = kb
    index_dir = tmp_path / "index"
    store_index(index_dir, documents, chunks)

    documents_path = index_dir / DOCUMENTS_FILE
    lines = documents_path.read_text(encoding="utf-8").splitlines()
    record: dict[str, object] = json.loads(lines[0])
    original_text = record["text"]
    assert isinstance(original_text, str)
    record["text"] = original_text + " (tampered)"
    lines[0] = json.dumps(record)
    documents_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_index(index_dir)


def test_load_index_rejects_missing_documents_jsonl(
    tmp_path: Path, kb: tuple[list[Document], list[Chunk]]
) -> None:
    documents, chunks = kb
    index_dir = tmp_path / "index"
    store_index(index_dir, documents, chunks)

    (index_dir / DOCUMENTS_FILE).unlink()

    with pytest.raises(ValueError):
        load_index(index_dir)


def test_save_index_rejects_documents_that_omit_chunks(
    tmp_path: Path, kb: tuple[list[Document], list[Chunk]]
) -> None:
    documents, chunks = kb
    assert len(documents) > 1
    index_dir = tmp_path / "index"
    dense, lexical = build_retrievers(chunks)

    with pytest.raises(ValueError):
        save_index(index_dir, dense, lexical, documents[:1])

    assert not (index_dir / MANIFEST_FILE).exists()


def test_failed_save_keeps_existing_index_loadable(
    tmp_path: Path, kb: tuple[list[Document], list[Chunk]]
) -> None:
    documents, chunks = kb
    dense, lexical = build_retrievers(chunks)
    directory = tmp_path / "index"
    save_index(directory, dense, lexical, documents)
    with pytest.raises(ValueError, match="duplicate document id"):
        save_index(directory, dense, lexical, [*documents, documents[0]])
    load_index(directory)
