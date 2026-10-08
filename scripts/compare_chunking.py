"""Compare chunking strategies on a corpus and print a Markdown table.

Usage:
    python scripts/compare_chunking.py PATH [--sizes 128 256] [--overlap-ratio 0.125]

Reports, per strategy and size: chunk count, token distribution, and the
fraction of chunks that cross a section boundary (structure-aware chunking
should keep this at or near zero; non-zero values come from merged tiny sections).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_engine.chunking import build_chunker, chunk_stats
from rag_engine.config import ChunkingConfig, ChunkStrategy
from rag_engine.ingestion import ingest_path
from rag_engine.models import Chunk, Document


def crosses_section(document: Document, chunk: Chunk) -> bool:
    first = document.section_at(chunk.start_char)
    last = document.section_at(chunk.end_char - 1)
    return first is not last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--overlap-ratio", type=float, default=0.125)
    args = parser.parse_args()

    documents = ingest_path(args.path).documents
    print(f"Corpus: {args.path} ({len(documents)} documents)\n")
    print("| strategy | size | overlap | chunks | mean tokens | p95 tokens | crossing sections |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for size in args.sizes:
        overlap = int(size * args.overlap_ratio)
        for strategy in ChunkStrategy:
            config = ChunkingConfig(strategy=strategy, chunk_size=size, chunk_overlap=overlap)
            chunker = build_chunker(config)
            chunks: list[Chunk] = []
            crossing = 0
            for document in documents:
                doc_chunks = chunker.chunk(document)
                crossing += sum(crosses_section(document, c) for c in doc_chunks)
                chunks.extend(doc_chunks)
            stats = chunk_stats(chunks)
            share = crossing / len(chunks) if chunks else 0.0
            print(
                f"| {strategy.value} | {size} | {overlap} | {stats['chunks']} | "
                f"{stats.get('tokens_mean', 0)} | {stats.get('tokens_p95', 0)} | {share:.0%} |"
            )


if __name__ == "__main__":
    main()
