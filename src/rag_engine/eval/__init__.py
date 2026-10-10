"""Evaluation harness: metrics, SciFact loader, structured-doc set, and runners."""

from rag_engine.eval.metrics import (
    METRIC_NAMES,
    Qrels,
    RankedList,
    aggregate_metrics,
    dedupe_preserve_order,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    relevant_ids,
    score_query,
)
from rag_engine.eval.runner import (
    EvalConfig,
    EvalReport,
    run_evaluation,
    run_scifact_eval,
    run_structured_eval,
    write_report,
)
from rag_engine.eval.scifact import (
    SCIFACT_MD5,
    SCIFACT_URL,
    SciFactSplit,
    download_scifact,
    ensure_scifact,
    load_scifact,
)
from rag_engine.eval.structured import StructuredDocSet, load_structured_docs

__all__ = [
    "METRIC_NAMES",
    "SCIFACT_MD5",
    "SCIFACT_URL",
    "EvalConfig",
    "EvalReport",
    "Qrels",
    "RankedList",
    "SciFactSplit",
    "StructuredDocSet",
    "aggregate_metrics",
    "dedupe_preserve_order",
    "download_scifact",
    "ensure_scifact",
    "hit_at_k",
    "load_scifact",
    "load_structured_docs",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "relevant_ids",
    "run_evaluation",
    "run_scifact_eval",
    "run_structured_eval",
    "score_query",
    "write_report",
]
