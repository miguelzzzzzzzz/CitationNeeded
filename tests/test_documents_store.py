from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.config import IngestionConfig
from rag_engine.documents import documents_path_for_chunks, read_documents, write_documents
from rag_engine.ingestion import ingest_path
from rag_engine.models import Document


def build(tmp_path: Path) -> list[Document]:
    root = tmp_path / "kb"
    (root / "guide").mkdir(parents=True)
    (root / "guide" / "intro.md").write_text(
        "# Intro\n\nThe first paragraph explains retrieval.\r\n\r\n## Setup\n\n"
        "Install the package and run the index command.\n",
        encoding="utf-8",
    )
    (root / "notes.txt").write_text("Plain notes about BM25 and dense vectors.\n", encoding="utf-8")
    report = ingest_path(root, IngestionConfig(corpus_id="handbook"))
    return report.documents


def lines_of(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_documents_round_trip(tmp_path: Path) -> None:
    docs = build(tmp_path)
    assert len(docs) == 2

    path = tmp_path / "documents.jsonl"
    write_documents(path, docs)

    assert path.exists()
    assert read_documents(path) == docs


def test_write_documents_rejects_duplicate_ids(tmp_path: Path) -> None:
    docs = build(tmp_path)
    path = tmp_path / "documents.jsonl"

    with pytest.raises(ValueError, match="duplicate document id"):
        write_documents(path, [docs[0], docs[0]])

    assert not path.exists()


def test_read_documents_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="documents file not found"):
        read_documents(tmp_path / "missing.jsonl")


def test_read_documents_rejects_invalid_lines(tmp_path: Path) -> None:
    docs = build(tmp_path)

    tampered = tmp_path / "tampered.jsonl"
    write_documents(tampered, [docs[0]])
    entry = json.loads(lines_of(tampered)[0])
    entry["text"] = entry["text"] + " tampered"
    tampered.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"invalid document on line 1"):
        read_documents(tampered)

    bad_id = tmp_path / "bad_id.jsonl"
    write_documents(bad_id, [docs[0]])
    entry = json.loads(lines_of(bad_id)[0])
    entry["doc_id"] = "not-a-matching-id"
    bad_id.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"invalid document on line 1"):
        read_documents(bad_id)


def test_read_documents_rejects_duplicate_lines(tmp_path: Path) -> None:
    docs = build(tmp_path)
    path = tmp_path / "documents.jsonl"
    write_documents(path, [docs[0]])

    line = lines_of(path)[0]
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate document id"):
        read_documents(path)


def test_read_documents_skips_blank_lines(tmp_path: Path) -> None:
    docs = build(tmp_path)
    path = tmp_path / "documents.jsonl"
    write_documents(path, docs)

    body = "\n" + "\n\n".join(lines_of(path)) + "\n\n"
    path.write_text(body, encoding="utf-8")

    assert read_documents(path) == docs


def test_documents_path_for_chunks() -> None:
    assert documents_path_for_chunks("out/chunks.jsonl") == Path("out/chunks.documents.jsonl")
