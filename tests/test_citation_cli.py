from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rag_engine.chunking import chunk_documents
from rag_engine.cli import main
from rag_engine.config import ChunkingConfig, IngestionConfig
from rag_engine.documents import (
    DOCUMENTS_FILE,
    documents_path_for_chunks,
    read_documents,
    verify_chunks,
)
from rag_engine.ingestion import ingest_path
from rag_engine.models import Chunk, Document
from rag_engine.retrieval.index_store import INDEX_FORMAT_VERSION, load_documents, load_index
from rag_engine.retrieval.retriever import LexicalRetriever


def make_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "kb"
    (root / "guide").mkdir(parents=True)
    (root / "guide" / "intro.md").write_text(
        "# Intro\n\nThe ﬁrst paragraph explains retrieval.\r\n\r\n"
        "## Setup\n\nInstall the package and run the index command.\n",
        encoding="utf-8",
    )
    (root / "notes.txt").write_text(
        "Plain notes about BM25 and dense vectors.\n",
        encoding="utf-8",
    )
    return root


def build(tmp_path: Path) -> tuple[list[Document], list[Chunk]]:
    root = make_corpus(tmp_path)
    report = ingest_path(root, IngestionConfig(corpus_id="handbook"))
    chunks = chunk_documents(report.documents, ChunkingConfig(chunk_size=16, chunk_overlap=4))
    return report.documents, chunks


def test_chunks_verify_against_their_documents(tmp_path: Path) -> None:
    documents, chunks = build(tmp_path)
    assert len(documents) == 2
    assert chunks
    verify_chunks(chunks, documents)
    known = {doc.doc_id for doc in documents}
    assert {chunk.doc_id for chunk in chunks} <= known


def test_lexical_hits_carry_verifiable_provenance(tmp_path: Path) -> None:
    documents, chunks = build(tmp_path)
    retriever = LexicalRetriever()
    retriever.index(chunks)
    hits = retriever.retrieve("retrieval", k=10)
    assert hits
    assert any("retrieval" in hit.chunk.text for hit in hits)
    by_id = {doc.doc_id: doc for doc in documents}
    for hit in hits:
        provenance = hit.provenance()
        document = by_id[hit.chunk.doc_id]
        assert provenance["content_hash"] == document.content_hash
        assert provenance["page_end"] == provenance["page"]
    top = hits[0].provenance()
    assert top["page"] is None
    assert top["page_end"] is None


def test_ingest_cli_writes_documents_beside_chunks(tmp_path: Path) -> None:
    root = make_corpus(tmp_path)
    out = tmp_path / "chunks.jsonl"
    assert main(["ingest", str(root), "--out", str(out)]) == 0
    documents_file = documents_path_for_chunks(out)
    assert documents_file == out.with_name("chunks.documents.jsonl")
    assert documents_file.is_file()
    stored = read_documents(documents_file)
    assert len(stored) == 2
    assert len({doc.doc_id for doc in stored}) == 2


def test_index_cli_writes_documents_into_the_index(tmp_path: Path) -> None:
    root = make_corpus(tmp_path)
    out = tmp_path / "chunks.jsonl"
    assert main(["ingest", str(root), "--out", str(out)]) == 0
    expected = [doc.doc_id for doc in read_documents(documents_path_for_chunks(out))]
    idx = tmp_path / "index"
    assert main(["index", str(out), "--out", str(idx), "--embedder", "hashing"]) == 0
    assert (idx / "index.json").is_file()
    assert (idx / DOCUMENTS_FILE).is_file()
    assert [doc.doc_id for doc in load_documents(idx)] == expected


def test_index_cli_asks_to_re_run_ingest_without_citation_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_corpus(tmp_path)
    out = tmp_path / "chunks.jsonl"
    assert main(["ingest", str(root), "--out", str(out)]) == 0
    documents_path_for_chunks(out).unlink()
    idx = tmp_path / "index"
    assert main(["index", str(out), "--out", str(idx), "--embedder", "hashing"]) == 2
    assert "re-run" in capsys.readouterr().err.lower()


def test_index_cli_rejects_edited_chunk_text(tmp_path: Path) -> None:
    root = make_corpus(tmp_path)
    out = tmp_path / "chunks.jsonl"
    assert main(["ingest", str(root), "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    target = next(i for i, line in enumerate(lines) if "retrieval" in line)
    record = json.loads(lines[target])
    key = next(k for k, v in record.items() if isinstance(v, str) and "retrieval" in v)
    text = record[key]
    match = re.search(r"[A-Za-z]+", text)
    assert match is not None
    start, end = match.span()
    record[key] = text[:start] + "z" * (end - start) + text[end:]
    assert len(record[key]) == len(text)
    lines[target] = json.dumps(record, ensure_ascii=False)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    idx = tmp_path / "index"
    assert main(["index", str(out), "--out", str(idx), "--embedder", "hashing"]) == 2


def test_load_index_rejects_a_legacy_format_version(tmp_path: Path) -> None:
    root = make_corpus(tmp_path)
    out = tmp_path / "chunks.jsonl"
    assert main(["ingest", str(root), "--out", str(out)]) == 0
    idx = tmp_path / "index"
    assert main(["index", str(out), "--out", str(idx), "--embedder", "hashing"]) == 0
    manifest_path = idx / "index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version_key = next(
        key for key, value in manifest.items() if str(value) == str(INDEX_FORMAT_VERSION)
    )
    manifest[version_key] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="no longer supported"):
        load_index(idx)
