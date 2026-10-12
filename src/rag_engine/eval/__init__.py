"""Evaluation harness: metrics, SciFact loader, structured-doc set, and runners."""

from rag_engine.eval.bootstrap import (
    BootstrapCI,
    load_per_query_scores,
    paired_bootstrap_ci,
)
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
    per_query_sidecar_path,
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
    "BootstrapCI",
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
    "load_per_query_scores",
    "load_scifact",
    "load_structured_docs",
    "mrr_at_k",
    "ndcg_at_k",
    "paired_bootstrap_ci",
    "per_query_sidecar_path",
    "recall_at_k",
    "relevant_ids",
    "run_evaluation",
    "run_scifact_eval",
    "run_structured_eval",
    "score_query",
    "write_report",
]
