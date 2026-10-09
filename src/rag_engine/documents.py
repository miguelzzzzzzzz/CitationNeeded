"""Persistence for the normalized documents that chunk offsets point into.

Loaders normalize newlines, ligatures and control characters before chunking, so
``Chunk.start_char``/``end_char`` are offsets into *normalized* text, not into the
raw file bytes. Persisting that normalized text next to the chunk file is what makes
citations verifiable: ``out/chunks.jsonl`` is always accompanied by
``out/chunks.documents.jsonl``, and :func:`verify_chunks` can re-check any chunk (or
later any citation built from a chunk) without re-running ingestion or depending on
the original files still being reachable.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from rag_engine.models import Chunk, Document

__all__ = [
    "DOCUMENTS_FILE",
    "documents_path_for_chunks",
    "read_documents",
    "verify_chunks",
    "write_documents",
]

DOCUMENTS_FILE = "documents.jsonl"


def write_documents(path: str | Path, documents: Iterable[Document]) -> int:
    """Write ``documents`` as JSON lines (UTF-8, ``\\n`` terminated) and return the count.

    Duplicate ``doc_id`` values are rejected before anything is written so a failed
    call cannot leave a partially written file behind.
    """
    document_path = Path(path)
    pending = list(documents)

    seen: set[str] = set()
    for document in pending:
        if document.doc_id in seen:
            raise ValueError(f"duplicate document id {document.doc_id!r} in {document_path}")
        seen.add(document.doc_id)

    document_path.parent.mkdir(parents=True, exist_ok=True)
    with document_path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in pending:
            handle.write(document.model_dump_json())
            handle.write("\n")
    return len(pending)


def read_documents(path: str | Path) -> list[Document]:
    """Load documents written by :func:`write_documents`.

    ``Document.model_validate_json`` re-runs the model validators, so a line whose
    ``doc_id``/``content_hash`` no longer matches its ``corpus_id``, ``source`` or
    ``text`` is rejected rather than silently trusted.
    """
    document_path = Path(path)
    if not document_path.is_file():
        raise ValueError(f"documents file not found: {document_path}")

    documents: list[Document] = []
    seen: set[str] = set()
    with document_path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                document = Document.model_validate_json(raw_line)
            except ValidationError as exc:
                raise ValueError(
                    f"invalid document on line {lineno} of {document_path}: {exc}"
                ) from exc
            if document.doc_id in seen:
                raise ValueError(
                    f"duplicate document id {document.doc_id!r} on line {lineno} of {document_path}"
                )
            seen.add(document.doc_id)
            documents.append(document)
    return documents


def documents_path_for_chunks(chunks_path: str | Path) -> Path:
    """Return the documents file that belongs to a chunks file.

    ``out/chunks.jsonl`` -> ``out/chunks.documents.jsonl``.
    """
    chunk_path = Path(chunks_path)
    return chunk_path.with_name(f"{chunk_path.stem}.{DOCUMENTS_FILE}")


def verify_chunks(chunks: Iterable[Chunk], documents: Iterable[Document]) -> None:
    """Re-check every chunk against the normalized documents it was derived from.

    Raises ``ValueError`` on the first problem found, naming the offending chunk, so
    an index can refuse to serve citations it cannot substantiate.
    """
    by_id: dict[str, Document] = {}
    for document in documents:
        if document.doc_id in by_id:
            raise ValueError(f"duplicate document id {document.doc_id!r}")
        by_id[document.doc_id] = document

    for chunk in chunks:
        found = by_id.get(chunk.doc_id)
        if found is None:
            raise ValueError(f"chunk {chunk.chunk_id} references unknown document {chunk.doc_id}")
        document = found

        recorded_source = chunk.metadata.get("source")
        if recorded_source is not None and recorded_source != document.source:
            raise ValueError(
                f"chunk {chunk.chunk_id} source {recorded_source!r} does not match "
                f"document {document.doc_id} source {document.source!r}"
            )

        recorded_hash = chunk.metadata.get("content_hash")
        if recorded_hash is not None and recorded_hash != document.content_hash:
            raise ValueError(
                f"chunk {chunk.chunk_id} content hash {recorded_hash!r} does not match "
                f"document {document.doc_id} hash {document.content_hash!r}"
            )

        if chunk.end_char > len(document.text):
            raise ValueError(
                f"chunk {chunk.chunk_id} end offset {chunk.end_char} exceeds text length "
                f"{len(document.text)} of document {document.doc_id}"
            )

        if chunk.text != document.text[chunk.start_char : chunk.end_char]:
            raise ValueError(
                f"chunk {chunk.chunk_id} text does not match document {document.doc_id} "
                f"at [{chunk.start_char}:{chunk.end_char}]"
            )
