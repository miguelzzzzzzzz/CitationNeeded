"""Command-line entry point.

Commands:

* ``ingest PATH --out FILE``: load documents, chunk them, write JSONL, print stats.
* ``index PATH --out INDEX_DIR``: build the dense and lexical indexes from a
  chunks JSONL file (or by ingesting documents) and save an index directory.
* ``search INDEX_DIR QUERY``: query a saved index directory with the dense,
  lexical, or hybrid retriever, optionally reranking the fused candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag_engine.chunking import chunk_documents, chunk_stats
from rag_engine.config import ChunkingConfig, IngestionConfig, Settings
from rag_engine.documents import (
    documents_path_for_chunks,
    read_documents,
    verify_chunks,
    write_documents,
)
from rag_engine.ingestion import ingest_path
from rag_engine.models import Chunk, Document
from rag_engine.retrieval import (
    CrossEncoderReranker,
    DenseRetriever,
    FusionMethod,
    HybridRetriever,
    LexicalRetriever,
    RerankingRetriever,
    RetrievedChunk,
    Retriever,
    embedder_from_spec,
    load_index,
    save_index,
)

_TEXT_PREVIEW_CHARS = 200

# Version of the ``search --json`` payload; bump on any breaking change to its
# keys. 1: {"schema_version", "query", "mode", "rerank", "hits": [hit, ...]}.
SEARCH_JSON_SCHEMA_VERSION = 1


def _add_corpus_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--corpus-id",
        default=None,
        help="Corpus namespace for document ids (default: RAG_CORPUS_ID, else root dir name).",
    )


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
    _add_corpus_arg(ingest)

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
    _add_corpus_arg(index)

    search = sub.add_parser("search", help="Query a saved index directory.")
    search.add_argument("index_dir", type=Path, help="Index directory written by 'index'.")
    search.add_argument("query", help="Query text.")
    search.add_argument(
        "--mode",
        choices=["dense", "lexical", "hybrid"],
        default="dense",
        help="Which retriever to query; 'hybrid' fuses the dense and lexical results.",
    )
    search.add_argument("-k", "--top-k", type=int, default=5, help="Number of hits to return.")
    search.add_argument(
        "--filter",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Metadata filter (repeatable; values for 'page' are parsed as int).",
    )
    # The fusion defaults are applied after parsing so that an explicit flag can
    # be distinguished from an omitted one and rejected outside hybrid mode.
    search.add_argument(
        "--fusion",
        choices=["rrf", "weighted"],
        default=None,
        help="Hybrid fusion method (default: rrf); requires --mode hybrid.",
    )
    search.add_argument(
        "--rrf-k",
        type=int,
        default=None,
        help="RRF constant k (default: 60); requires --mode hybrid.",
    )
    search.add_argument(
        "--weight",
        action="append",
        default=None,
        metavar="NAME=FLOAT",
        help="Hybrid weight for 'dense' or 'lexical' (repeatable); requires --mode hybrid.",
    )
    # Defaulted to None so an explicit --candidates can be told apart from the
    # omitted flag; the effective default of 50 is applied after parsing.
    search.add_argument(
        "--candidates",
        type=int,
        default=None,
        help=(
            "Candidate pool size for fusion and reranking (default: 50); "
            "requires --mode hybrid or --rerank."
        ),
    )
    search.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank the retrieved candidates with a cross-encoder.",
    )
    search.add_argument(
        "--rerank-model",
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model used by --rerank.",
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
    if getattr(args, "corpus_id", None) is not None:
        try:
            ingestion = IngestionConfig.model_validate(
                settings.ingestion.model_dump() | {"corpus_id": args.corpus_id}
            )
        except ValidationError as exc:
            raise ValueError(f"invalid --corpus-id: {exc}") from exc
        settings = settings.model_copy(update={"ingestion": ingestion})
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

    # Verify before writing anything, then write the documents first: a chunks
    # file never exists without the normalized text its offsets point into.
    verify_chunks(chunks, report.documents)
    documents_path = documents_path_for_chunks(args.out)
    write_documents(documents_path, report.documents)
    with args.out.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")

    stats = {
        "documents": len(report.documents),
        "skipped": len(report.skipped),
        "strategy": settings.chunking.strategy.value,
        "chunk_size": settings.chunking.chunk_size,
        "chunk_overlap": settings.chunking.chunk_overlap,
        "documents_file": str(documents_path),
        "corpus_id": report.corpus_id,
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

    ``documents`` always holds every ingested document, including those that
    produced no chunks, so citation offsets can be resolved and verified.
    """

    chunks: list[Chunk]
    documents: list[Document]
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
        chunks = _read_chunks_file(path)
        documents_path = documents_path_for_chunks(path)
        if not documents_path.is_file():
            raise ValueError(
                f"documents file not found: {documents_path} "
                "(re-run `rag-engine ingest`, which writes it next to the chunks file)"
            )
        documents = read_documents(documents_path)
        # Reject a stale or hand-edited chunks file instead of indexing text
        # whose citations would not resolve against the documents.
        verify_chunks(chunks, documents)
        return _IndexInput(chunks=chunks, documents=documents, skipped=0)
    report = ingest_path(path, settings.ingestion)
    if report.skipped:
        print("\nSkipped:", file=sys.stderr)
        for item in report.skipped:
            print(f"  {item.source}: {item.reason}", file=sys.stderr)
    chunks = chunk_documents(report.documents, settings.chunking)
    return _IndexInput(
        chunks=chunks,
        documents=report.documents,
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

    try:
        # Constructing the retriever reads embedder.dimension, which is what
        # lazily loads the embedding model, so it must stay under this guard.
        dense = DenseRetriever(embedder)
        lexical = LexicalRetriever()
        dense.index(loaded.chunks)
        # Reading the dimension materialises the embedding model lazily, so it
        # belongs here: failing before save_index avoids writing a broken index.
        dimension = embedder.dimension
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    lexical.index(loaded.chunks)
    save_index(args.out, dense, lexical, loaded.documents)

    summary = {
        "documents": len(loaded.documents),
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


def _parse_weight_overrides(values: list[str] | None) -> dict[str, float]:
    """Parse repeated ``NAME=FLOAT`` flags; the source name must be dense or lexical."""
    weights: dict[str, float] = {}
    for raw in values or []:
        name, separator, value = raw.partition("=")
        if not separator or name not in {"dense", "lexical"}:
            raise ValueError(
                f"invalid weight {raw!r}: expected NAME=FLOAT with NAME in dense|lexical"
            )
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"invalid weight {raw!r}: expected a number") from exc
        # NaN and infinities would poison every fused score, so reject them here.
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError(f"invalid weight {raw!r}: expected a finite value >= 0")
        weights[name] = parsed
    return weights


def _format_hit(hit: RetrievedChunk) -> str:
    """One summary line: rank, score, citation location, components, and chunk id."""
    chunk = hit.chunk
    source = chunk.metadata.get("source")
    location: str = str(source) if source else chunk.doc_id
    if chunk.heading_path:
        location += " > " + " > ".join(chunk.heading_path)
    if chunk.page is not None:
        location += f" (p. {chunk.page})"
    if hit.components:
        rendered = ", ".join(f"{name}={value:.4f}" for name, value in hit.components.items())
        location += f"  ({rendered})"
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

    candidates = 50 if args.candidates is None else args.candidates
    try:
        if candidates < 1:
            raise ValueError("candidates must be positive")
        # Rejected outside hybrid/rerank because the pool size would only be
        # forwarded to a retriever that ignores it.
        if args.candidates is not None and args.mode != "hybrid" and not args.rerank:
            raise ValueError("--candidates requires --mode hybrid or --rerank")
        if args.mode != "hybrid" and any(
            value is not None for value in (args.fusion, args.rrf_k, args.weight)
        ):
            raise ValueError("--fusion/--rrf-k/--weight require --mode hybrid")
        method: FusionMethod | str = args.fusion or "rrf"
        if args.rrf_k is not None and args.fusion == "weighted":
            raise ValueError("--rrf-k requires --fusion rrf")
        rrf_k = 60 if args.rrf_k is None else args.rrf_k
        if rrf_k < 0:
            raise ValueError("rrf-k must be >= 0")
        parsed_weights = _parse_weight_overrides(args.weight)
        # Zero-weight sources still contribute their ranks (or nothing under
        # weighted fusion), so an all-zero map would silently disable fusion.
        if parsed_weights and all(value == 0 for value in parsed_weights.values()):
            raise ValueError("at least one weight must be positive")
        weights: Mapping[str, float] | None = parsed_weights or None
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

    retriever: Retriever = dense
    try:
        # Both the hybrid retriever and the reranker validate their arguments in
        # their constructors, so they share one ValueError -> exit 2 handler.
        if args.mode == "lexical":
            retriever = lexical
        elif args.mode == "hybrid":
            retriever = HybridRetriever(
                [dense, lexical],
                method=method,
                weights=weights,
                rrf_k=rrf_k,
                candidates=candidates,
            )
        if args.rerank:
            # The cross-encoder loads lazily on the first retrieve call, so a
            # missing fastembed surfaces as an ImportError below, like the embedder.
            retriever = RerankingRetriever(
                retriever,
                CrossEncoderReranker(
                    args.rerank_model,
                    cache_dir=settings.embedding.cache_dir,
                    threads=settings.embedding.threads,
                ),
                candidates=candidates,
            )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        hits = retriever.retrieve(args.query, args.top_k, filters or None)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            "schema_version": SEARCH_JSON_SCHEMA_VERSION,
            "query": args.query,
            "mode": args.mode,
            "rerank": bool(args.rerank),
            "hits": [hit.to_dict() for hit in hits],
        }
        # Scores are finite by construction; allow_nan=False keeps the output strict JSON.
        print(json.dumps(payload, indent=2, allow_nan=False))
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
