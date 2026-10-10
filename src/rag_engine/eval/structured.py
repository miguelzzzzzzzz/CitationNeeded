"""Loader for the in-repo structured-document eval set (milestone M4).

``evals/structured`` holds hand-written, multi-section Markdown documents plus
binary qrels. It is a *structure* benchmark: it exists to show whether heading
paths survive chunking and are usable for retrieval, so scores here are only
meaningful relative to other runs on the same set. It is deliberately not a
SciFact replacement and must not be reported as one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rag_engine.ingestion.loaders import MarkdownLoader
from rag_engine.models import Document

__all__ = ["StructuredDocSet", "StructuredQuery", "load_structured_docs"]

DOCS_DIRNAME = "docs"
QRELS_FILENAME = "qrels.jsonl"


class StructuredQuery(BaseModel):
    """A single eval query and its binary relevance labels.

    ``relevant_docs`` holds document stems as written in ``qrels.jsonl``;
    ``relevant_sections`` optionally gives, per relevant doc, the heading path of
    the section that actually answers the query. It is empty for doc-level qrels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    text: str
    relevant_docs: tuple[str, ...]
    relevant_sections: tuple[tuple[str, ...], ...] = ()


class StructuredDocSet(BaseModel):
    """An in-memory view of one structured eval root."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path
    documents: dict[str, Document]
    queries: dict[str, StructuredQuery]

    def stem_to_doc_id(self) -> dict[str, str]:
        """Map qrels file stems to the stable ``Document.doc_id`` values."""
        return {stem: document.doc_id for stem, document in self.documents.items()}

    def qrels_binary(self) -> dict[str, dict[str, int]]:
        """Binary qrels keyed by doc id, as consumed by the eval runners.

        Stems are resolved through the loaded documents (never hashed here), so
        the ids always match what ingestion produced for this ``corpus_id``.
        """
        doc_ids = self.stem_to_doc_id()
        return {
            query_id: {doc_ids[stem]: 1 for stem in query.relevant_docs}
            for query_id, query in self.queries.items()
        }


def load_structured_docs(root: Path, *, corpus_id: str = "structured-eval") -> StructuredDocSet:
    """Load ``root/docs/*.md`` and ``root/qrels.jsonl`` into a doc set.

    ``corpus_id`` is passed straight to the Markdown loader so doc ids stay
    stable across checkouts and machines.
    """
    docs_dir = root / DOCS_DIRNAME
    qrels_path = root / QRELS_FILENAME
    if not docs_dir.is_dir():
        raise ValueError(f"{root}: expected a {DOCS_DIRNAME}/ directory of Markdown documents")
    if not qrels_path.is_file():
        raise ValueError(f"{root}: expected a {QRELS_FILENAME} file")

    loader = MarkdownLoader()
    documents: dict[str, Document] = {}
    for path in sorted(docs_dir.glob("*.md")):
        # ``source`` must be a normalized POSIX path relative to the corpus
        # root: it is hashed into the doc id, so this keeps ids identical no
        # matter where the eval set lives on disk.
        source = f"{DOCS_DIRNAME}/{path.name}"
        documents[path.stem] = loader.load(path, source, corpus_id=corpus_id)

    queries = _load_queries(qrels_path, documents)
    return StructuredDocSet(root=root, documents=documents, queries=queries)


def _load_queries(
    qrels_path: Path, documents: Mapping[str, Document]
) -> dict[str, StructuredQuery]:
    """Parse qrels.jsonl, rejecting duplicate ids and unknown document stems."""
    queries: dict[str, StructuredQuery] = {}
    missing: set[str] = set()
    lines = qrels_path.read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload: object = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{qrels_path}:{lineno}: invalid JSON: {exc.msg}") from exc
        query = _parse_query(payload, qrels_path, lineno)
        if query.query_id in queries:
            raise ValueError(f"{qrels_path}:{lineno}: duplicate query_id {query.query_id!r}")
        queries[query.query_id] = query
        missing.update(stem for stem in query.relevant_docs if stem not in documents)
    if missing:
        listed = ", ".join(sorted(missing))
        raise ValueError(f"{qrels_path}: qrels reference unknown document stems: {listed}")
    return queries


def _parse_query(payload: object, path: Path, lineno: int) -> StructuredQuery:
    where = f"{path}:{lineno}"
    if not isinstance(payload, dict):
        raise ValueError(f"{where}: expected a JSON object, got {type(payload).__name__}")

    query_id = _required_str(payload, "query_id", where)
    text = _required_str(payload, "text", where)

    raw_docs = payload.get("relevant_docs")
    if not isinstance(raw_docs, list):
        raise ValueError(f"{where}: 'relevant_docs' must be a list of file stems")
    relevant_docs = _str_tuple(raw_docs, "relevant_docs", where)

    relevant_sections = _parse_sections(payload.get("relevant_sections"), where)
    if relevant_sections and len(relevant_sections) != len(relevant_docs):
        raise ValueError(
            f"{where}: 'relevant_sections' has {len(relevant_sections)} entries "
            f"but 'relevant_docs' has {len(relevant_docs)}"
        )

    return StructuredQuery(
        query_id=query_id,
        text=text,
        relevant_docs=relevant_docs,
        relevant_sections=relevant_sections,
    )


def _required_str(payload: Mapping[str, Any], key: str, where: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: {key!r} must be a non-empty string")
    return value


def _str_tuple(values: list[Any], field_name: str, where: str) -> tuple[str, ...]:
    result: list[str] = []
    for entry in values:
        if not isinstance(entry, str) or not entry:
            raise ValueError(f"{where}: {field_name!r} entries must be non-empty strings")
        result.append(entry)
    return tuple(result)


def _parse_sections(value: object, where: str) -> tuple[tuple[str, ...], ...]:
    """Normalize ``relevant_sections`` to a tuple of heading paths.

    Only the aligned shape (a list of heading-path lists) is accepted; an empty
    inner list marks doc-level relevance for that position.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{where}: 'relevant_sections' must be a list of heading paths")
    sections: list[tuple[str, ...]] = []
    for entry in value:
        if not isinstance(entry, list):
            raise ValueError(f"{where}: each 'relevant_sections' entry must be a list of headings")
        sections.append(_str_tuple(entry, "relevant_sections", where))
    return tuple(sections)
