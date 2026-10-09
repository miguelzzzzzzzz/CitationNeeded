"""Tests for corpus-scoped document ids and chunk offset validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rag_engine.chunking import chunk_documents
from rag_engine.cli import main
from rag_engine.config import ChunkingConfig, IngestionConfig, Settings
from rag_engine.ingestion import ingest_path
from rag_engine.models import (
    DEFAULT_CORPUS_ID,
    Chunk,
    Document,
    corpus_id_from_name,
    make_doc_id,
    validate_corpus_id,
)

_MARKDOWN = "# Guide\n\nWidgets ship in crates. Gadgets ship in boxes. Both are counted.\n"

# sha256("handbook:guide/intro.md")[:16]; recomputed in the test below.
_PINNED_DOC_ID = "120aa960849299ee"


def _write_markdown(root: Path, relative: str, text: str = _MARKDOWN) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _document() -> Document:
    return Document.create(
        source="guide/intro.md",
        title="Intro",
        format="markdown",
        text=_MARKDOWN,
        corpus_id="handbook",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_make_doc_id_is_pinned_and_matches_sha256_scheme() -> None:
    expected = hashlib.sha256(b"handbook:guide/intro.md").hexdigest()[:16]
    assert expected == _PINNED_DOC_ID
    assert make_doc_id("handbook", "guide/intro.md") == _PINNED_DOC_ID

    with pytest.raises(ValueError, match="corpus_id must match"):
        make_doc_id("Bad:ID", "guide/intro.md")


@pytest.mark.parametrize(
    "corpus_id",
    ["handbook", "corpus-1", "a.b_c", "a" * 64, DEFAULT_CORPUS_ID],
)
def test_validate_corpus_id_accepts_valid_ids(corpus_id: str) -> None:
    assert validate_corpus_id(corpus_id) == corpus_id


@pytest.mark.parametrize(
    "corpus_id",
    ["", "Handbook", "a:b", "-x", "a" * 65, "a b", "ünïcode"],
)
def test_validate_corpus_id_rejects_invalid_ids(corpus_id: str) -> None:
    with pytest.raises(ValueError, match="corpus_id must match"):
        validate_corpus_id(corpus_id)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("My Docs", "my-docs"),
        ("docs", "docs"),
        ("!!!", DEFAULT_CORPUS_ID),
        ("__x__", "x__"),
        ("Team Notes!!", "team-notes"),
        ("Version 2.0", "version-2.0"),
        ("a" * 100, "a" * 64),
    ],
)
def test_corpus_id_from_name_slugifies_directory_names(name: str, expected: str) -> None:
    assert corpus_id_from_name(name) == expected


@pytest.mark.parametrize(
    "name",
    ["My Docs", "!!!", "__x__", "   ", "café", "9lives", "-x", "文档", "a" * 100],
)
def test_corpus_id_from_name_always_returns_a_valid_id(name: str) -> None:
    slug = corpus_id_from_name(name)
    assert validate_corpus_id(slug) == slug


def test_doc_ids_are_stable_across_ingest_roots(tmp_path: Path) -> None:
    roots = [tmp_path / "a" / "kb", tmp_path / "b" / "elsewhere"]
    for root in roots:
        _write_markdown(root, "guide/intro.md")

    config = IngestionConfig(corpus_id="handbook")
    first = ingest_path(roots[0], config=config)
    second = ingest_path(roots[1], config=config)

    doc_id = make_doc_id("handbook", "guide/intro.md")
    assert [(doc.source, doc.doc_id) for doc in first.documents] == [("guide/intro.md", doc_id)]
    assert [(doc.source, doc.doc_id) for doc in second.documents] == [("guide/intro.md", doc_id)]
    assert first.corpus_id == second.corpus_id == "handbook"

    chunking = ChunkingConfig()
    first_chunks = chunk_documents(first.documents, chunking)
    second_chunks = chunk_documents(second.documents, chunking)

    assert first_chunks
    assert [chunk.chunk_id for chunk in first_chunks] == [chunk.chunk_id for chunk in second_chunks]
    assert {chunk.doc_id for chunk in first_chunks} == {doc_id}
    assert first_chunks[0].metadata["corpus_id"] == "handbook"
    assert first_chunks[0].metadata["content_hash"] == first.documents[0].content_hash


def test_default_corpus_id_comes_from_root_directory_name(tmp_path: Path) -> None:
    root = tmp_path / "Team Notes"
    _write_markdown(root, "guide/intro.md")

    report = ingest_path(root, config=IngestionConfig())

    assert report.corpus_id == "team-notes"
    assert [(doc.corpus_id, doc.doc_id) for doc in report.documents] == [
        ("team-notes", make_doc_id("team-notes", "guide/intro.md"))
    ]


def test_same_relative_path_in_different_corpora_does_not_collide(tmp_path: Path) -> None:
    alpha_root = tmp_path / "alpha-root"
    beta_root = tmp_path / "beta-root"
    for root in (alpha_root, beta_root):
        _write_markdown(root, "guide/intro.md")

    alpha = ingest_path(alpha_root, config=IngestionConfig(corpus_id="alpha"))
    beta = ingest_path(beta_root, config=IngestionConfig(corpus_id="beta"))

    assert alpha.documents[0].source == beta.documents[0].source == "guide/intro.md"
    assert alpha.documents[0].doc_id != beta.documents[0].doc_id

    alpha_chunks = chunk_documents(alpha.documents, ChunkingConfig())
    beta_chunks = chunk_documents(beta.documents, ChunkingConfig())
    assert alpha_chunks
    assert beta_chunks
    assert {chunk.chunk_id for chunk in alpha_chunks}.isdisjoint(
        {chunk.chunk_id for chunk in beta_chunks}
    )


def test_document_create_derives_doc_id_and_content_hash() -> None:
    doc = _document()
    assert doc.corpus_id == "handbook"
    assert doc.doc_id == make_doc_id("handbook", "guide/intro.md")
    assert doc.content_hash == hashlib.sha256(_MARKDOWN.encode("utf-8")).hexdigest()


def test_document_rejects_tampered_doc_id() -> None:
    doc = _document()
    with pytest.raises(ValidationError, match="doc_id"):
        Document.model_validate({**doc.model_dump(), "doc_id": "0" * 16})


def test_document_rejects_tampered_text() -> None:
    doc = _document()
    tampered = _MARKDOWN.replace("Widgets", "Widgits")
    assert tampered != _MARKDOWN
    assert len(tampered) == len(_MARKDOWN)
    with pytest.raises(ValidationError, match="content_hash"):
        Document.model_validate({**doc.model_dump(), "text": tampered})


def test_document_rejects_invalid_corpus_id() -> None:
    doc = _document()
    with pytest.raises(ValidationError, match="corpus_id"):
        Document.model_validate({**doc.model_dump(), "corpus_id": "Bad:ID"})


def test_chunk_accepts_valid_offsets() -> None:
    chunk = Chunk(
        chunk_id="c0",
        doc_id=_document().doc_id,
        index=0,
        text="Widget",
        start_char=4,
        end_char=10,
        token_count=1,
    )
    assert len(chunk.text) == chunk.end_char - chunk.start_char == 6


@pytest.mark.parametrize(
    ("text", "start_char", "end_char", "message"),
    [
        ("", 3, 3, "end_char"),
        ("Widget", 0, 10, "text length"),
        ("Widgets", 0, 6, "text length"),
    ],
)
def test_chunk_rejects_inconsistent_offsets(
    text: str, start_char: int, end_char: int, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Chunk(
            chunk_id="c0",
            doc_id="0" * 16,
            index=0,
            text=text,
            start_char=start_char,
            end_char=end_char,
            token_count=1,
        )


def test_settings_reads_corpus_id_from_env() -> None:
    settings = Settings.from_env({"RAG_CORPUS_ID": "kb"})
    assert settings.ingestion.corpus_id == "kb"

    with pytest.raises(ValueError):
        Settings.from_env({"RAG_CORPUS_ID": "Bad:ID"})


def test_cli_ingest_uses_corpus_id(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    _write_markdown(root, "guide/intro.md")
    out = tmp_path / "chunks.jsonl"

    main(["ingest", str(root), "--out", str(out), "--corpus-id", "handbook"])

    entries = _read_jsonl(out)
    assert entries
    assert {entry["doc_id"] for entry in entries} == {make_doc_id("handbook", "guide/intro.md")}


def test_cli_ingest_rejects_invalid_corpus_id(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    _write_markdown(root, "guide/intro.md")
    out = tmp_path / "chunks.jsonl"

    assert main(["ingest", str(root), "--out", str(out), "--corpus-id", "Bad:ID"]) == 2
    assert not out.exists()


@pytest.mark.parametrize(
    "source", ["", "./a.md", "a\\b.md", "/a.md", "a//b.md", "../a.md", "a/./b.md"]
)
def test_make_doc_id_rejects_unnormalized_sources(source: str) -> None:
    with pytest.raises(ValueError, match="normalized relative POSIX path"):
        make_doc_id("handbook", source)


def test_front_matter_cannot_shadow_provenance_metadata() -> None:
    doc = Document.create(
        source="a.md",
        title="A",
        format="markdown",
        text="Alpha body text for the chunk.",
        corpus_id="handbook",
        metadata={"corpus_id": "other", "source": "b.md", "content_hash": "x"},
    )
    (chunk,) = chunk_documents([doc], ChunkingConfig())
    assert chunk.metadata["corpus_id"] == "handbook"
    assert chunk.metadata["source"] == "a.md"
    assert chunk.metadata["content_hash"] == doc.content_hash
