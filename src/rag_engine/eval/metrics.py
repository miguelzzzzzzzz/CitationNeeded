"""Retrieval ranking metrics for document-level evaluation (milestone M4).

Metrics here operate on *document-level* rankings: a query is credited when a
retrieved chunk belongs to a relevant document, so a chunk-level pipeline is
scored by mapping hits to their ``doc_id`` first.  Duplicate ``doc_id`` values
are collapsed by keeping the *first* (i.e. best) rank, which is the usual way to
evaluate chunked corpora without letting one document occupy several slots.

Graded relevance comes from ``qrels`` (``doc_id -> int >= 0``, where ``0`` means
non-relevant); documents missing from ``qrels`` are treated as grade ``0``.

Degenerate inputs are well defined: an empty (or fully non-overlapping) ranked
list scores ``0.0``, and a query with no relevant documents also scores ``0.0``
for every metric.  ``ndcg@k`` returns ``0.0`` when the ideal DCG is ``0``, i.e.
when no graded relevance exists for the query.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# Ordered ``doc_id`` list; rank 1 is the first element.  Duplicates are ignored
# by keeping the first occurrence.
RankedList = Sequence[str]

# Relevance grades: ``doc_id -> int >= 0``.  Missing docs are treated as 0.
Qrels = Mapping[str, int]

METRIC_NAMES: tuple[str, ...] = (
    "recall@1",
    "recall@5",
    "recall@10",
    "hit@1",
    "hit@5",
    "hit@10",
    "mrr@10",
    "ndcg@10",
)

__all__ = [
    "METRIC_NAMES",
    "Qrels",
    "RankedList",
    "aggregate_metrics",
    "dedupe_preserve_order",
    "hit_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "relevant_ids",
    "score_query",
]


def _validate_k(k: int) -> None:
    """Reject non-positive cut-offs before any metric arithmetic happens."""
    if k <= 0:
        raise ValueError("k must be positive")


def dedupe_preserve_order(ids: Sequence[str]) -> list[str]:
    """Return ``ids`` with duplicates removed, keeping the first occurrence.

    The first occurrence carries the best rank, so later duplicates of the same
    document must not consume additional ranking slots.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for doc_id in ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


def relevant_ids(qrels: Qrels, *, min_relevance: int = 1) -> frozenset[str]:
    """Documents whose grade is at least ``min_relevance``."""
    return frozenset(doc_id for doc_id, grade in qrels.items() if grade >= min_relevance)


def recall_at_k(
    ranked: RankedList,
    qrels: Qrels,
    k: int = 10,
    *,
    min_relevance: int = 1,
) -> float:
    """Fraction of relevant documents found in the top ``k`` unique ranks."""
    _validate_k(k)
    relevant = relevant_ids(qrels, min_relevance=min_relevance)
    if not relevant:
        return 0.0
    top_k = set(dedupe_preserve_order(ranked)[:k])
    return len(relevant & top_k) / len(relevant)


def hit_at_k(
    ranked: RankedList,
    qrels: Qrels,
    k: int = 10,
    *,
    min_relevance: int = 1,
) -> float:
    """``1.0`` if any relevant document appears in the top ``k``, else ``0.0``."""
    _validate_k(k)
    relevant = relevant_ids(qrels, min_relevance=min_relevance)
    if not relevant:
        return 0.0
    return 1.0 if any(doc_id in relevant for doc_id in dedupe_preserve_order(ranked)[:k]) else 0.0


def mrr_at_k(
    ranked: RankedList,
    qrels: Qrels,
    k: int = 10,
    *,
    min_relevance: int = 1,
) -> float:
    """Reciprocal rank (1-based, after dedupe) of the first relevant document."""
    _validate_k(k)
    relevant = relevant_ids(qrels, min_relevance=min_relevance)
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(dedupe_preserve_order(ranked)[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: RankedList, qrels: Qrels, k: int = 10) -> float:
    """Normalised DCG over graded relevance, using ``2**rel - 1`` gains.

    The ranking is truncated to ``k`` unique documents (missing positions count
    as grade 0, which contributes no gain) and normalised by the ideal DCG built
    from the top-``k`` positive grades in ``qrels``.
    """
    _validate_k(k)
    unique_ranked = dedupe_preserve_order(ranked)[:k]
    dcg = math.fsum(
        (2.0 ** qrels.get(doc_id, 0) - 1.0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(unique_ranked, start=1)
    )
    ideal_grades = sorted((grade for grade in qrels.values() if grade > 0), reverse=True)[:k]
    idcg = math.fsum(
        (2.0**grade - 1.0) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1)
    )
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def aggregate_metrics(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Mean of each metric key across queries, skipping queries that lack it.

    Keys are emitted in first-seen order; a key present in no query is omitted.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in per_query.values():
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    return {name: total / counts[name] for name, total in totals.items()}


def score_query(
    ranked: RankedList,
    qrels: Qrels,
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute recall/hit at every ``k`` in ``ks`` plus MRR and nDCG at ``max(ks)``.

    Labels follow ``"{metric}@{k}"`` (for example ``"recall@5"``).  ``ks`` must
    be non-empty because MRR/nDCG cut-offs are derived from its maximum.
    """
    if not ks:
        raise ValueError("ks must be non-empty")
    largest = max(ks)
    scores: dict[str, float] = {}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(ranked, qrels, k)
        scores[f"hit@{k}"] = hit_at_k(ranked, qrels, k)
    scores[f"mrr@{largest}"] = mrr_at_k(ranked, qrels, largest)
    scores[f"ndcg@{largest}"] = ndcg_at_k(ranked, qrels, largest)
    return scores
