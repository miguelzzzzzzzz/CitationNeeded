"""Evaluation runner: build an in-memory index, retrieve, score, report (M4).

A single :func:`run_evaluation` call chunks the corpus, builds the requested
retrievers, indexes the chunks and scores every labelled query. Nothing is
persisted between runs beyond the JSON report, so a report is fully determined
by its dataset plus the :class:`EvalConfig` embedded in it. The dataset-specific
wrappers only load data and decide where the report lands.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from rag_engine.chunking import chunk_documents
from rag_engine.config import ChunkingConfig, ChunkStrategy
from rag_engine.eval.metrics import (
    aggregate_metrics,
    dedupe_preserve_order,
    relevant_ids,
    score_query,
)
from rag_engine.eval.scifact import SciFactSplit, ensure_scifact
from rag_engine.eval.structured import load_structured_docs
from rag_engine.models import Document, Section
from rag_engine.retrieval import (
    CrossEncoderReranker,
    DenseRetriever,
    FusionMethod,
    HybridRetriever,
    LexicalRetriever,
    RerankingRetriever,
    Retriever,
)
from rag_engine.retrieval.index_store import embedder_from_spec
from rag_engine.retrieval.retriever import RetrievedChunk

__all__ = [
    "EvalConfig",
    "EvalReport",
    "build_scifact_documents",
    "doc_ranked_list",
    "git_dirty",
    "git_sha",
    "hardware_note",
    "retrieve_documents",
    "run_evaluation",
    "run_scifact_eval",
    "run_structured_eval",
    "write_report",
]

_ALLOWED_DATASETS: Final[frozenset[str]] = frozenset({"scifact", "structured"})
_ALLOWED_MODES: Final[frozenset[str]] = frozenset({"dense", "lexical", "hybrid", "hybrid+rerank"})
_RERANK_MODE: Final[str] = "hybrid+rerank"
_DEFAULT_RESULTS_DIR: Final[Path] = Path("evals/results")
_WARMUP_QUERY: Final[str] = "warm-up query for model loading"
_RERANK_EXTRA_HINT: Final[str] = (
    "reranking requires the optional embeddings extra, e.g. "
    "`pip install 'production-rag-engine[embeddings]'`"
)


class EvalConfig(BaseModel):
    """Every knob that can change an evaluation outcome, minus the dataset."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    split: str = "test"
    modes: tuple[str, ...] = ("dense", "lexical", "hybrid")
    embedder: str = "hashing"
    fusion: str = "rrf"
    rrf_k: int = 60
    candidates: int = 50
    top_k: int = 10
    depth: int = 50
    max_depth: int | None = None
    chunk_strategy: str = "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 64
    corpus_id: str = "eval"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    @field_validator("dataset")
    @classmethod
    def _check_dataset(cls, value: str) -> str:
        if value not in _ALLOWED_DATASETS:
            raise ValueError(f"dataset must be one of {sorted(_ALLOWED_DATASETS)}, got {value!r}")
        return value

    @field_validator("modes")
    @classmethod
    def _check_modes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("modes must not be empty")
        unknown = [mode for mode in value if mode not in _ALLOWED_MODES]
        if unknown:
            raise ValueError(f"unknown modes {unknown}; allowed: {sorted(_ALLOWED_MODES)}")
        return value

    @field_validator("chunk_strategy")
    @classmethod
    def _check_chunk_strategy(cls, value: str) -> str:
        allowed = {strategy.value for strategy in ChunkStrategy}
        if value not in allowed:
            raise ValueError(f"chunk_strategy must be one of {sorted(allowed)}, got {value!r}")
        return value

    @field_validator("depth")
    @classmethod
    def _check_depth(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"depth must be >= 1, got {value}")
        return value

    @field_validator("max_depth")
    @classmethod
    def _check_max_depth(cls, value: int | None, info: ValidationInfo) -> int | None:
        depth = info.data.get("depth")
        if value is not None and isinstance(depth, int) and value < depth:
            raise ValueError(f"max_depth ({value}) must be >= depth ({depth})")
        return value


class EvalReport(BaseModel):
    """Machine-readable outcome of one evaluation run.

    Field guide:

    * ``n_queries`` counts scored queries only. Queries with no (or an empty) qrels entry
      are counted in
      ``n_queries_unlabelled`` and queries whose qrels hold no grade >= 1 in
      ``n_queries_skipped_no_relevant``; both are skipped rather than scored as 0.
    * ``retrieval[mode]`` holds ``initial_depth``, the ``max_depth`` cap,
      ``max_depth_used``/``mean_depth_used`` and ``queries_short_of_k`` -- the number of
      queries that ended with fewer than ``top_k`` unique documents because the index
      ran out of hits or the ``max_depth`` cap was reached.
    * ``git_dirty`` is the working-tree state next to ``git_sha``; None outside a checkout.
    * ``smoke_only`` marks toy runs that are wiring checks, not quality claims.
    """

    schema_version: int = 2
    created_at: str
    git_sha: str
    git_dirty: bool | None = None
    hardware: dict[str, str | int]
    config: EvalConfig
    dataset_checksum: str
    dataset_name: str
    n_documents: int
    n_queries: int
    n_queries_unlabelled: int = 0
    n_queries_skipped_no_relevant: int = 0
    n_chunks: int
    metrics: dict[str, dict[str, float]]
    retrieval: dict[str, dict[str, float]]
    latency_ms: dict[str, dict[str, float]]
    notes: str = ""
    smoke_only: bool = False


def hardware_note() -> dict[str, str | int]:
    """Host description for the report; values stay JSON-friendly."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 0,
    }


def git_sha(cwd: Path | None = None) -> str:
    """Commit the run was produced from, or "" outside a usable checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd if cwd is not None else Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def git_dirty(cwd: Path | None = None) -> bool | None:
    """Whether tracked files differ from HEAD; None outside a usable checkout.

    ``--untracked-files=no`` keeps stray build artefacts from flagging every run.
    """
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd if cwd is not None else Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def doc_ranked_list(hits: Sequence[RetrievedChunk]) -> list[str]:
    """Document ids in retrieval order; the first (best) rank wins per document."""
    return dedupe_preserve_order([hit.chunk.doc_id for hit in hits])


def build_scifact_documents(split_data: SciFactSplit, *, corpus_id: str) -> list[Document]:
    """Turn SciFact corpus records into whole-abstract :class:`Document`s."""
    documents: list[Document] = []
    for record in split_data.corpus.values():
        text = f"{record.title}\n\n{record.text}" if record.title else record.text
        documents.append(
            Document.create(
                source=f"scifact/{record.doc_id}.txt",
                title=record.title,
                format="txt",
                text=text,
                corpus_id=corpus_id,
                sections=(Section(start_char=0, end_char=len(text)),),
                metadata={"beir_doc_id": record.doc_id},
            )
        )
    return documents


def _fingerprint(
    documents: Sequence[Document],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
) -> str:
    """sha256 over the evaluated content, for datasets shipping no checksum."""
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.doc_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(document.text.encode("utf-8"))
        digest.update(b"\x1e")
    for query_id in sorted(queries):
        digest.update(query_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(queries[query_id].encode("utf-8"))
        digest.update(b"\x1f")
        for doc_id in sorted(qrels.get(query_id, {})):
            digest.update(f"{doc_id}:{qrels[query_id][doc_id]}".encode())
            digest.update(b"\x1e")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    """Hex sha256 of ``path``, or "" when it is missing or unreadable."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile: exact for n == 1, no interpolation for small n."""
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered))
    return ordered[min(max(rank - 1, 0), len(ordered) - 1)]


def _build_reranker(config: EvalConfig) -> CrossEncoderReranker:
    """Construct the cross-encoder, surfacing a missing extra as a clear error."""
    try:
        return CrossEncoderReranker(config.rerank_model)
    except ImportError as exc:
        raise RuntimeError(_RERANK_EXTRA_HINT) from exc


def _hybrid(config: EvalConfig, dense: Retriever, lexical: Retriever) -> HybridRetriever:
    return HybridRetriever(
        [dense, lexical],
        method=FusionMethod(config.fusion),
        rrf_k=config.rrf_k,
        candidates=config.candidates,
    )


def _build_retrievers(
    config: EvalConfig, dense: Retriever, lexical: Retriever
) -> dict[str, Retriever]:
    """One retriever per requested mode; the reranker only exists if asked for."""
    retrievers: dict[str, Retriever] = {}
    for mode in config.modes:
        if mode == "dense":
            retrievers[mode] = dense
        elif mode == "lexical":
            retrievers[mode] = lexical
        elif mode == "hybrid":
            retrievers[mode] = _hybrid(config, dense, lexical)
        else:  # "hybrid+rerank"; EvalConfig validation leaves no other option
            retrievers[mode] = RerankingRetriever(
                _hybrid(config, dense, lexical),
                _build_reranker(config),
                candidates=config.candidates,
            )
    return retrievers


def _retrieve_hits(
    retriever: Retriever, query: str, k: int, *, reranking: bool
) -> Sequence[RetrievedChunk]:
    """Retrieve; fastembed is imported lazily, so rerank failures become RuntimeError."""
    try:
        return retriever.retrieve(query, k=k)
    except ImportError as exc:
        if not reranking:
            raise
        raise RuntimeError(_RERANK_EXTRA_HINT) from exc


def retrieve_documents(
    retriever: Retriever,
    query: str,
    *,
    k: int,
    depth: int,
    max_depth: int,
    reranking: bool,
) -> tuple[list[str], int, bool]:
    """Fetch up to ``k`` unique documents, doubling the chunk depth while more may exist.

    Hybrid fusion collapses hits to unique documents, so a fixed chunk depth can leave a
    mode with fewer than ``k`` documents while another mode sees more; starting every mode
    at the same depth and over-fetching keeps the comparison fair.

    Returns the ranked document ids (cut to ``k``), the final depth fetched, and whether
    the retriever ran out of hits before ``k`` unique documents were found.
    """
    hits = _retrieve_hits(retriever, query, depth, reranking=reranking)
    docs = doc_ranked_list(hits)
    while len(docs) < k and len(hits) == depth and depth < max_depth:
        depth = min(depth * 2, max_depth)
        hits = _retrieve_hits(retriever, query, depth, reranking=reranking)
        docs = doc_ranked_list(hits)
    return docs[:k], depth, len(docs) < k


def run_evaluation(
    *,
    documents: Sequence[Document],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    config: EvalConfig,
    dataset_checksum: str,
    dataset_name: str,
    notes: str = "",
    smoke_only: bool = False,
) -> EvalReport:
    """Chunk, index and score ``queries``/``qrels`` in memory, then report.

    Every mode fetches the same initial chunk depth and over-fetches until it holds
    ``top_k`` unique documents or exhausts the index, so no mode is scored on fewer
    documents than another.
    """
    chunks = chunk_documents(
        documents,
        ChunkingConfig(
            strategy=ChunkStrategy(config.chunk_strategy),
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        ),
    )
    dense = DenseRetriever(embedder_from_spec(config.embedder))
    lexical = LexicalRetriever()
    dense.index(chunks)
    lexical.index(chunks)

    # Partition queries up front: unlabelled and all-non-relevant queries are skipped, not
    # scored as 0, and the counts are reported separately from the scored set.
    labelled: dict[str, Mapping[str, int]] = {}
    n_unlabelled = 0
    n_no_relevant = 0
    for query_id in sorted(queries):
        relevant = qrels.get(query_id)
        if not relevant:  # no qrels entry, or an empty one
            n_unlabelled += 1
        elif not relevant_ids(relevant):
            n_no_relevant += 1
        else:
            labelled[query_id] = relevant

    # ``cap`` mirrors the configured ceiling, falling back to the corpus size; the outer
    # max(..., 1) keeps an empty chunk set from producing a depth of 0.
    cap = max(config.max_depth or len(chunks), 1)
    initial = max(1, min(max(config.depth, config.top_k), cap))

    metrics: dict[str, dict[str, float]] = {}
    retrieval: dict[str, dict[str, float]] = {}
    latency: dict[str, dict[str, float]] = {}
    for mode, retriever in _build_retrievers(config, dense, lexical).items():
        reranking = mode == _RERANK_MODE
        if labelled:
            # Untimed warm-up: loads lazy models before the first timed query. It uses a
            # fixed text that is not an evaluated query, so no scored query is pre-cached.
            retrieve_documents(
                retriever,
                _WARMUP_QUERY,
                k=config.top_k,
                depth=initial,
                max_depth=cap,
                reranking=reranking,
            )
        scored: dict[str, dict[str, float]] = {}
        timings: list[float] = []
        depths: list[int] = []
        short_of_k = 0
        for query_id, relevant in labelled.items():
            started = time.perf_counter()
            docs, depth_used, exhausted = retrieve_documents(
                retriever,
                queries[query_id],
                k=config.top_k,
                depth=initial,
                max_depth=cap,
                reranking=reranking,
            )
            timings.append((time.perf_counter() - started) * 1000.0)
            scored[query_id] = score_query(docs, relevant)
            depths.append(depth_used)
            if exhausted:
                short_of_k += 1
        metrics[mode] = aggregate_metrics(scored) if scored else {}
        latency[mode] = (
            {
                "retrieve_p50": _percentile(timings, 0.5),
                "retrieve_p95": _percentile(timings, 0.95),
            }
            if timings
            else {}
        )
        retrieval[mode] = {
            "initial_depth": float(initial),
            "max_depth": float(cap),
            "max_depth_used": float(max(depths)) if depths else 0.0,
            "mean_depth_used": (sum(depths) / len(depths)) if depths else 0.0,
            "queries_short_of_k": float(short_of_k),
        }

    return EvalReport(
        created_at=datetime.now(UTC).isoformat(),
        git_sha=git_sha(),
        git_dirty=git_dirty(),
        hardware=hardware_note(),
        config=config,
        dataset_checksum=dataset_checksum,
        dataset_name=dataset_name,
        n_documents=len(documents),
        n_queries=len(labelled),
        n_queries_unlabelled=n_unlabelled,
        n_queries_skipped_no_relevant=n_no_relevant,
        n_chunks=len(chunks),
        metrics=metrics,
        retrieval=retrieval,
        latency_ms=latency,
        notes=notes,
        smoke_only=smoke_only,
    )


def write_report(report: EvalReport, path: Path) -> Path:
    """Write ``report`` as pretty, strict JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.model_dump(mode="json"), indent=2, allow_nan=False)
    path.write_text(f"{payload}\n", encoding="utf-8")
    return path


def _report_path(results_dir: Path | None, name: str) -> Path:
    base = results_dir if results_dir is not None else _DEFAULT_RESULTS_DIR
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return base / f"{name}_{stamp}.json"


def run_scifact_eval(
    dest_dir: Path,
    *,
    config: EvalConfig | None = None,
    results_dir: Path | None = None,
    url: str | None = None,
    expected_md5: str | None = None,
) -> EvalReport:
    """Fetch SciFact into ``dest_dir`` if needed, evaluate it and write the report."""
    from rag_engine.eval.scifact import SCIFACT_MD5, SCIFACT_URL

    cfg = config if config is not None else EvalConfig(dataset="scifact")
    if cfg.dataset != "scifact":
        raise ValueError(f"run_scifact_eval expects dataset='scifact', got {cfg.dataset!r}")
    split_data = ensure_scifact(
        dest_dir,
        split=cfg.split,
        url=url if url is not None else SCIFACT_URL,
        expected_md5=expected_md5 if expected_md5 is not None else SCIFACT_MD5,
    )
    documents = build_scifact_documents(split_data, corpus_id=cfg.corpus_id)
    queries = {qid: query.text for qid, query in split_data.queries.items()}
    # SciFact qrels key documents by BEIR ids; our Document.doc_id is hashed.
    beir_to_doc: dict[str, str] = {}
    for doc in documents:
        beir_id = doc.metadata.get("beir_doc_id")
        if not isinstance(beir_id, str) or not beir_id:
            raise ValueError(f"document {doc.doc_id!r} missing string metadata beir_doc_id")
        beir_to_doc[beir_id] = doc.doc_id
    referenced = {beir_id for grades in split_data.qrels.values() for beir_id in grades}
    missing = referenced - beir_to_doc.keys()
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(
            f"{len(missing)} qrels beir_doc_ids absent from corpus (examples: {sample})"
        )
    qrels = {
        query_id: {beir_to_doc[beir_id]: grade for beir_id, grade in grades.items()}
        for query_id, grades in split_data.qrels.items()
    }
    report = run_evaluation(
        documents=documents,
        queries=queries,
        qrels=qrels,
        config=cfg,
        dataset_checksum=split_data.checksum or _fingerprint(documents, queries, qrels),
        dataset_name=f"scifact/{cfg.split}",
    )
    write_report(report, _report_path(results_dir, f"scifact_{cfg.split}_{cfg.embedder}"))
    return report


def run_structured_eval(
    root: Path,
    *,
    config: EvalConfig | None = None,
    results_dir: Path | None = None,
) -> EvalReport:
    """Evaluate the local structured dataset rooted at ``root``.

    The bundled set is a wiring fixture, so the report is flagged ``smoke_only`` and must
    not be read as a quality claim.
    """
    cfg = (
        config
        if config is not None
        else EvalConfig(dataset="structured", chunk_strategy=ChunkStrategy.STRUCTURE.value)
    )
    if cfg.dataset != "structured":
        raise ValueError(f"run_structured_eval expects dataset='structured', got {cfg.dataset!r}")
    docset = load_structured_docs(root, corpus_id=cfg.corpus_id)
    documents = list(docset.documents.values())
    queries = {qid: query.text for qid, query in docset.queries.items()}
    qrels = docset.qrels_binary()
    report = run_evaluation(
        documents=documents,
        queries=queries,
        qrels=qrels,
        config=cfg,
        dataset_checksum=_file_sha256(root / "qrels.jsonl")
        or _fingerprint(documents, queries, qrels),
        dataset_name="structured",
        notes=(
            "smoke test only: 3 documents / 8 queries, too small for quality claims; "
            "scored at document level (relevant_sections not used)"
        ),
        smoke_only=True,
    )
    write_report(report, _report_path(results_dir, f"structured_{cfg.embedder}"))
    return report
