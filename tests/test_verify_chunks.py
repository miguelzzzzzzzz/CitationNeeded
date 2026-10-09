from __future__ import annotations

from pathlib import Path

import pytest

from rag_engine.chunking import chunk_documents
from rag_engine.config import ChunkingConfig, IngestionConfig
from rag_engine.documents import verify_chunks
from rag_engine.ingestion import ingest_path
from rag_engine.models import Chunk, Document

Corpus = tuple[list[Document], list[Chunk]]


def _build(tmp_path: Path) -> Corpus:
    root = tmp_path / "kb"
    (root / "guide").mkdir(parents=True)
    (root / "guide" / "intro.md").write_text(
        "# Intro\n\nThe ﬁrst paragraph explains retrieval.\r\n\r\n"
        "## Setup\n\nInstall the package and run the index command.\n",
        encoding="utf-8",
    )
    (root / "notes.txt").write_text("Plain notes about BM25 and dense vectors.\n", encoding="utf-8")
    report = ingest_path(root, IngestionConfig(corpus_id="handbook"))
    chunks = chunk_documents(report.documents, ChunkingConfig(chunk_size=16, chunk_overlap=4))
    return report.documents, chunks


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    return _build(tmp_path)


def _documents_by_id(documents: list[Document]) -> dict[str, Document]:
    return {document.doc_id: document for document in documents}


def test_verify_chunks_accepts_pipeline_output(corpus: Corpus) -> None:
    documents, chunks = corpus
    assert len(documents) == 2
    assert len(chunks) >= 2
    verify_chunks(chunks, documents)


def test_every_chunk_text_is_the_exact_document_slice(corpus: Corpus) -> None:
    documents, chunks = corpus
    by_id = _documents_by_id(documents)
    for chunk in chunks:
        document = by_id[chunk.doc_id]
        assert 0 <= chunk.start_char < chunk.end_char <= len(document.text)
        assert chunk.text == document.text[chunk.start_char : chunk.end_char]


def test_verify_chunks_rejects_unknown_doc_id(corpus: Corpus) -> None:
    documents, chunks = corpus
    forged = chunks[0].model_copy(update={"doc_id": "handbook:missing.md"})
    assert forged.doc_id != chunks[0].doc_id
    with pytest.raises(ValueError, match="unknown document"):
        verify_chunks([forged], documents)


def test_verify_chunks_rejects_content_hash_mismatch(corpus: Corpus) -> None:
    documents, chunks = corpus
    original = chunks[0]
    forged = original.model_copy(
        update={"metadata": {**original.metadata, "content_hash": "0" * 64}}
    )
    assert forged.metadata["content_hash"] != original.metadata["content_hash"]
    with pytest.raises(ValueError, match="content hash"):
        verify_chunks([forged], documents)


def test_verify_chunks_rejects_source_mismatch(corpus: Corpus) -> None:
    documents, chunks = corpus
    original = chunks[0]
    forged = original.model_copy(
        update={"metadata": {**original.metadata, "source": "guide/elsewhere.md"}}
    )
    assert forged.metadata["source"] != original.metadata["source"]
    with pytest.raises(ValueError, match="source"):
        verify_chunks([forged], documents)


def test_verify_chunks_rejects_end_char_past_document_end(corpus: Corpus) -> None:
    documents, chunks = corpus
    original = chunks[0]
    document = _documents_by_id(documents)[original.doc_id]
    forged = original.model_copy(update={"end_char": len(document.text) + 1})
    assert forged.end_char > len(document.text)
    with pytest.raises(ValueError, match="exceeds text length"):
        verify_chunks([forged], documents)


def test_verify_chunks_rejects_text_outside_the_slice(corpus: Corpus) -> None:
    documents, chunks = corpus
    original = chunks[0]
    document = _documents_by_id(documents)[original.doc_id]
    forged = original.model_copy(update={"text": f"{original.text}?"})
    assert forged.text != document.text[forged.start_char : forged.end_char]
    with pytest.raises(ValueError, match="text does not match"):
        verify_chunks([forged], documents)


def test_verify_chunks_rejects_duplicate_documents(tmp_path: Path) -> None:
    documents, chunks = _build(tmp_path)
    with pytest.raises(ValueError, match="duplicate document id"):
        verify_chunks(chunks, [*documents, documents[0]])
