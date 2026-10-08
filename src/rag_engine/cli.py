"""Command-line entry point.

Current commands:

* ``ingest PATH --out FILE``: load documents, chunk them, write JSONL, print stats.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from rag_engine.chunking import chunk_documents, chunk_stats
from rag_engine.config import ChunkingConfig, Settings
from rag_engine.ingestion import ingest_path


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
    ingest.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "structure"],
        default=None,
        help="Override RAG_CHUNK_STRATEGY.",
    )
    ingest.add_argument("--chunk-size", type=int, default=None, help="Override RAG_CHUNK_SIZE.")
    ingest.add_argument(
        "--chunk-overlap", type=int, default=None, help="Override RAG_CHUNK_OVERLAP."
    )
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
        settings = Settings(ingestion=settings.ingestion, chunking=chunking)
    return settings


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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        return cmd_ingest(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
