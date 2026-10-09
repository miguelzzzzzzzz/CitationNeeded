"""Command-line entry point.

Commands:

* ``ingest PATH --out FILE``: load documents, chunk them, write JSONL, print stats.
* ``index PATH --out INDEX_DIR``: build the dense and lexical indexes from a
  chunks JSONL file (or by ingesting documents) and save an index directory.
* ``search INDEX_DIR QUERY``: query a saved index directory in dense or lexical mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag_engine.chunking import chunk_documents, chunk_stats
from rag_engine.config import ChunkingConfig, Settings
from rag_engine.ingestion import ingest_path
from rag_engine.models import Chunk
from rag_engine.retrieval import (
    DenseRetriever,
    LexicalRetriever,
    RetrievedChunk,
    Retriever,
    embedder_from_spec,
    load_index,
    save_index,
)

_TEXT_PREVIEW_CHARS = 200


def _add_chunking_args(parser: argparse.ArgumentParser) -> None:
    """Add the chunking overrides shared by ``ingest`` and ``index``."""
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "structure"],
        default=None,
        help="Override RAG_CHUNK_STRATEGY.",
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="Override RAG_CHUNK_SIZE.")
    parser.add_argument(
        "--chunk-overlap", type=int, default=None, help="Override RAG_CHUNK_OVERLAP."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-engine",
        description="Production-style RAG engine CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a file or directory into chunks.")
    ingest.add_argument("path", type=Path, help="File or directory to ingest.")
    ingest.add_argument(
        "--out",
        type=Path,
        required=True,
        help="JSONL file to write chunks into (one JSON object per line).",
    )
    _add_chunking_args(ingest)

    index = sub.add_parser("index", help="Build a dense + lexical index.")
    index.add_argument(
        "path",
        type=Path,
        help=(
            "A chunks .jsonl file produced by 'ingest', or a document "
            "file/directory (.md, .txt, .html, .pdf)."
        ),
    )
    index.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Index directory to write (dense/, lexical/, index.json).",
    )
    index.add_argument(
        "--embedder",
        default="fastembed",
        help="Embedder spec, e.g. 'hashing', 'hashing-256', 'fastembed:BAAI/bge-small-en-v1.5'.",
    )
    _add_chunking_args(index)

    search = sub.add_parser("search", help="Query a saved index directory.")
    search.add_argument("index_dir", type=Path, help="Index directory written by 'index'.")
    search.add_argument("query", help="Query text.")
    search.add_argument(
        "--mode",
        choices=["dense", "lexical"],
        default="dense",
        help="Which retriever to query.",
    )
    search.add_argument("-k", "--top-k", type=int, default=5, help="Number of hits to return.")
    search.add_argument(
        "--filter",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Metadata filter (repeatable; values for 'page' are parsed as int).",
    )
    search.add_argument("--json", action="store_true", help="Print hits as JSON.")
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    updates: dict[str, object] = {}
    if args.strategy is not None:
        updates["strategy"] = args.strategy
    if args.chunk_size is not None:
        updates["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        updates["chunk_overlap"] = args.chunk_overlap
    if updates:
        # Re-validate instead of model_copy(update=...), which skips validation
        # and would accept e.g. chunk_overlap >= chunk_size.
        try:
            chunking = ChunkingConfig.model_validate(settings.chunking.model_dump() | updates)
        except ValidationError as exc:
            raise ValueError(f"invalid chunking options: {exc}") from exc
        # model_copy is safe here: `chunking` was validated above. Keeps the
        # ingestion and embedding sections from the environment intact.
        settings = settings.model_copy(update={"chunking": chunking})
    return settings


def _is_chunks_file(path: Path) -> bool:
    """True for the JSONL files written by ``ingest``; such a file is never raw input."""
    return path.is_file() and path.suffix == ".jsonl"


def _has_chunking_overrides(args: argparse.Namespace) -> bool:
    """True when any chunking override flag was given on the command line."""
    return (
        args.strategy is not None or args.chunk_size is not None or args.chunk_overlap is not None
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    try:
        settings = _settings_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.path.exists():
        print(f"error: path does not exist: {args.path}", file=sys.stderr)
        return 2
    report = ingest_path(args.path, settings.ingestion)
    chunks = chunk_documents(report.documents, settings.chunking)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")

    stats = {
        "documents": len(report.documents),
        "skipped": len(report.skipped),
        "strategy": settings.chunking.strategy.value,
        "chunk_size": settings.chunking.chunk_size,
        "chunk_overlap": settings.chunking.chunk_overlap,
        **chunk_stats(chunks),
    }
    print(json.dumps(stats, indent=2))
    if report.skipped:
        print("\nSkipped:", file=sys.stderr)
        for item in report.skipped:
            print(f"  {item.source}: {item.reason}", file=sys.stderr)
    return 0


@dataclass(frozen=True)
class _IndexInput:
    """Chunks to index plus the source statistics echoed by ``index``.

    ``documents`` is ``None`` when ``PATH`` was a chunks file, since that file
    no longer carries the document count it was produced from.
    """

    chunks: list[Chunk]
    documents: int | None
    skipped: int


def _read_chunks_file(path: Path) -> list[Chunk]:
    """Parse one ``Chunk`` per non-blank line, naming the offending line on failure."""
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                chunks.append(Chunk.model_validate_json(stripped))
            except ValidationError as exc:
                raise ValueError(f"invalid chunk on line {lineno} of {path}: {exc}") from exc
    return chunks


def _load_index_chunks(path: Path, settings: Settings) -> _IndexInput:
    """Read chunks from a JSONL file, or ingest and chunk documents from ``path``.

    A ``.jsonl`` file is assumed to hold ``Chunk`` objects (the ``ingest``
    output); anything else goes through the same code path as ``ingest`` so both
    commands chunk documents identically.
    """
    if _is_chunks_file(path):
        return _IndexInput(chunks=_read_chunks_file(path), documents=None, skipped=0)
    report = ingest_path(path, settings.ingestion)
    if report.skipped:
        print("\nSkipped:", file=sys.stderr)
        for item in report.skipped:
            print(f"  {item.source}: {item.reason}", file=sys.stderr)
    chunks = chunk_documents(report.documents, settings.chunking)
    return _IndexInput(
        chunks=chunks,
        documents=len(report.documents),
        skipped=len(report.skipped),
    )


def cmd_index(args: argparse.Namespace) -> int:
    try:
        settings = _settings_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.path.exists():
        print(f"error: path does not exist: {args.path}", file=sys.stderr)
        return 2
    if _is_chunks_file(args.path) and _has_chunking_overrides(args):
        # Pre-chunked input is indexed verbatim, so the flags have no effect.
        print(
            "warning: chunking options are ignored when indexing a chunks file",
            file=sys.stderr,
        )
    try:
        embedder = embedder_from_spec(args.embedder, settings.embedding)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        loaded = _load_index_chunks(args.path, settings)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not loaded.chunks:
        print("error: no chunks to index", file=sys.stderr)
        return 1

    dense = DenseRetriever(embedder)
    lexical = LexicalRetriever()
    try:
        dense.index(loaded.chunks)
        # Reading the dimension materialises the embedding model lazily, so it
        # belongs here: failing before save_index avoids writing a broken index.
        dimension = embedder.dimension
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    lexical.index(loaded.chunks)
    save_index(args.out, dense, lexical)

    summary = {
        "documents": loaded.documents,
        "skipped": loaded.skipped,
        "chunks": len(loaded.chunks),
        "embedder": embedder.name,
        "dimension": dimension,
        "index_dir": str(args.out),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _parse_filters(values: list[str] | None) -> dict[str, Any]:
    """Turn repeated ``KEY=VALUE`` flags into vector-store filters.

    A repeated key accumulates into a list, which the store reads as "any of";
    ``page`` is the only key coerced to ``int``.
    """
    filters: dict[str, Any] = {}
    for raw in values or []:
        key, separator, value = raw.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid filter {raw!r}: expected KEY=VALUE")
        parsed: Any
        if key == "page":
            try:
                parsed = int(value)
            except ValueError as exc:
                raise ValueError(f"invalid page filter {raw!r}: expected an integer") from exc
        else:
            parsed = value
        if key in filters:
            existing = filters[key]
            if isinstance(existing, list):
                filters[key] = [*existing, parsed]
            else:
                filters[key] = [existing, parsed]
        else:
            filters[key] = parsed
    return filters


def _format_hit(hit: RetrievedChunk) -> str:
    """One summary line: rank, score, citation location, and chunk id."""
    chunk = hit.chunk
    source = chunk.metadata.get("source")
    location: str = str(source) if source else chunk.doc_id
    if chunk.heading_path:
        location += " > " + " > ".join(chunk.heading_path)
    if chunk.page is not None:
        location += f" (p. {chunk.page})"
    return f"{hit.rank}. {hit.score:.4f}  {location}  [{chunk.chunk_id}]"


def _format_text(text: str) -> str:
    """Indented single-line preview of the chunk text, truncated when long."""
    collapsed = " ".join(text.split())
    if len(collapsed) > _TEXT_PREVIEW_CHARS:
        collapsed = collapsed[:_TEXT_PREVIEW_CHARS] + "..."
    return f"    {collapsed}"


def cmd_search(args: argparse.Namespace) -> int:
    if args.top_k <= 0:
        print("error: k must be positive", file=sys.stderr)
        return 2
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        filters = _parse_filters(args.filter)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.index_dir.exists():
        print(f"error: index directory does not exist: {args.index_dir}", file=sys.stderr)
        return 2
    try:
        dense, lexical = load_index(args.index_dir, settings.embedding)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    retriever: Retriever = dense if args.mode == "dense" else lexical
    try:
        hits = retriever.retrieve(args.query, args.top_k, filters or None)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([hit.to_dict() for hit in hits], indent=2))
        return 0
    if not hits:
        print("no results", file=sys.stderr)
        return 0
    for hit in hits:
        print(_format_hit(hit))
        print(_format_text(hit.chunk.text))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        return cmd_ingest(args)
    if args.command == "index":
        return cmd_index(args)
    if args.command == "search":
        return cmd_search(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
