"""Box-only CLI for the M5 answer eval on SciFact.

Running this script without ``--replay`` calls a paid/quota LLM API and writes
the raw replies into a committed JSONL cache; it is therefore never executed in
CI. Committed reports cite the model id and the run date (see ``created_at``),
and ``--replay`` regenerates the metrics without any API call: it still loads
SciFact and rebuilds retrieval (the cache is keyed on the exact contexts sent).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag_engine.chunking import chunk_documents
from rag_engine.config import ChunkingConfig, ChunkStrategy
from rag_engine.eval.answer_cache import CachedAnswerGenerator, CacheMissError, ResponseCache
from rag_engine.eval.answer_runner import (
    AnswerEvalRecord,
    build_answer_report,
    build_items,
    pick_subsets,
    run_answer_eval,
    write_answer_report,
)
from rag_engine.eval.runner import build_scifact_documents, git_dirty, git_sha
from rag_engine.eval.scifact import SCIFACT_MD5, SCIFACT_URL, ensure_scifact
from rag_engine.models import Chunk
from rag_engine.retrieval import DenseRetriever, FusionMethod, HybridRetriever, LexicalRetriever
from rag_engine.retrieval.index_store import embedder_from_spec


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="answer_eval",
        description="Record (paid LLM API) or replay the SciFact test answer eval.",
    )
    parser.add_argument("--n-answerable", type=int, default=20)
    parser.add_argument("--n-no-relevant", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=5, help="contexts per item")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--embedder", default="fastembed")
    parser.add_argument("--dataset-dir", type=Path, default=Path("evals/datasets"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("evals/answers/scifact_test_deepseek-v4.1-flash.cache.jsonl"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("evals/results"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--base-url", default="https://api.cline.bot/api/v1")
    parser.add_argument("--model", default="cline-pass/deepseek-v4.1-flash")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--max-repair-attempts", type=int, default=1)
    parser.add_argument(
        "--call-log",
        type=Path,
        default=None,
        help="append one JSON line per item (api calls made, error) to this file",
    )
    return parser.parse_args(argv)


def _build_retriever(chunks: list[Chunk], embedder: str) -> HybridRetriever:
    dense = DenseRetriever(embedder_from_spec(embedder))
    lexical = LexicalRetriever()
    dense.index(chunks)
    lexical.index(chunks)
    return HybridRetriever([dense, lexical], method=FusionMethod("rrf"), rrf_k=60, candidates=50)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    api_key = os.environ.get("RAG_LLM_API_KEY") or os.environ.get("CLINE_API_KEY")
    if not args.replay and not api_key:
        print(
            "error: set RAG_LLM_API_KEY or CLINE_API_KEY, or pass --replay",
            file=sys.stderr,
        )
        return 2

    print(f"loading SciFact test split into {args.dataset_dir}", file=sys.stderr)
    split = ensure_scifact(
        args.dataset_dir, split="test", url=SCIFACT_URL, expected_md5=SCIFACT_MD5
    )
    documents = build_scifact_documents(split, corpus_id="scifact")
    chunks = chunk_documents(
        documents,
        ChunkingConfig(strategy=ChunkStrategy("fixed"), chunk_size=512, chunk_overlap=64),
    )
    print(f"indexing {len(chunks)} chunks ({args.embedder})", file=sys.stderr)
    retriever = _build_retriever(chunks, args.embedder)

    beir_to_doc = {doc.metadata["beir_doc_id"]: doc.doc_id for doc in documents}
    gold: dict[str, set[str]] = {}
    for qid, rels in split.qrels.items():
        if qid not in split.queries:
            continue
        doc_ids = {
            beir_to_doc[beir] for beir, grade in rels.items() if grade >= 1 and beir in beir_to_doc
        }
        if doc_ids:
            gold[qid] = doc_ids
    labelled_ids = sorted(gold)
    claims = {qid: split.queries[qid].text for qid in labelled_ids}
    print(f"{len(labelled_ids)} labelled queries", file=sys.stderr)

    answerable_ids, no_relevant_ids = pick_subsets(
        labelled_ids,
        n_answerable=args.n_answerable,
        n_no_relevant=args.n_no_relevant,
        seed=args.seed,
    )
    items = build_items(
        retriever=retriever,
        claims=claims,
        gold=gold,
        answerable_ids=answerable_ids,
        no_relevant_ids=no_relevant_ids,
        k=args.k,
        depth=args.depth,
    )
    total = len(items)
    print(f"{total} eval items", file=sys.stderr)

    generator = CachedAnswerGenerator(
        cache=ResponseCache(args.cache),
        replay_only=args.replay,
        base_url=args.base_url,
        model=args.model,
        api_key=None if args.replay else api_key,
        timeout_s=args.timeout,
        max_tokens=args.max_tokens,
        temperature=0.0,
        max_repair_attempts=args.max_repair_attempts,
    )

    api_calls_seen = 0

    def on_item(i: int, record: AnswerEvalRecord) -> None:
        nonlocal api_calls_seen
        grade = record.grade
        if args.call_log is not None:
            entry = {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "purpose": f"answer eval item {record.query_id} ({grade.kind})",
                "model": args.model,
                "api_calls": generator.n_api_calls - api_calls_seen,
                "status": "failed" if record.error else "ok",
                "error": record.error,
            }
            with args.call_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        api_calls_seen = generator.n_api_calls
        print(
            f"[{i + 1}/{total}] {record.query_id} {grade.kind} "
            f"abstained={grade.abstained} "
            f"valid={grade.n_valid_citations}/{grade.n_raw_citations} "
            f"error={grade.error}",
            file=sys.stderr,
        )

    try:
        records = run_answer_eval(
            items, generator, max_sentences=args.max_sentences, on_item=on_item
        )
    except CacheMissError as exc:
        print(f"replay cache miss: {exc}", file=sys.stderr)
        return 3

    report = build_answer_report(
        records,
        config={
            "model": args.model,
            "base_url": args.base_url,
            "n_answerable": args.n_answerable,
            "n_no_relevant": args.n_no_relevant,
            "seed": args.seed,
            "k": args.k,
            "depth": args.depth,
            "embedder": args.embedder,
            "fusion": "rrf",
            "rrf_k": 60,
            "candidates": 50,
            "chunking": {"strategy": "fixed", "chunk_size": 512, "chunk_overlap": 64},
            "corpus_id": "scifact",
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "max_sentences": args.max_sentences,
            "max_repair_attempts": args.max_repair_attempts,
            "replay": bool(args.replay),
            "cache": args.cache.as_posix(),
        },
        created_at=datetime.now(UTC).isoformat(),
        git_sha=git_sha(),
        git_dirty=git_dirty(),
        dataset_name="scifact/test",
        dataset_checksum=split.checksum,
        n_cache_hits=generator.n_cache_hits,
        n_api_calls=generator.n_api_calls,
    )

    out_path = args.out
    if out_path is None and not args.replay:
        slug = args.model.replace("/", "_")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = args.results_dir / f"answers_scifact_test_{slug}_{stamp}.json"
    if out_path is not None:
        write_answer_report(report, out_path)
        print(f"wrote {out_path}", file=sys.stderr)

    print(json.dumps(report["metrics"], indent=2))
    print(
        f"cache hits={generator.n_cache_hits} api calls={generator.n_api_calls}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
