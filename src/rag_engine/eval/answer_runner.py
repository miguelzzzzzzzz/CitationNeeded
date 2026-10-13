"""Orchestration for the M5 answer eval: subset picking, item building, replayable runs.

Retrieval and generation are injected, so unit tests use fakes. A run records each
item's claim (CC BY 4.0), its context ids and text hashes (abstract text is not
copied into reports), every raw model reply, the parsed response and a code-only
grade. ``no_relevant`` items are a gold-removal ablation: the query's qrels-gold
documents are dropped from its retrieved contexts, so abstaining is expected (qrels
can be incomplete, so a few such items may still be answerable).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rag_engine.answering.models import AnswerResponse
from rag_engine.eval.answer_cache import CachedAnswerGenerator, CacheMissError
from rag_engine.eval.answers import (
    ANSWER_EVAL_SCHEMA_VERSION,
    SELECTION_METHOD,
    AnswerGrade,
    EvalContext,
    ItemKind,
    aggregate_grades,
    contexts_from_hits,
    drop_gold_hits,
    grade_answer,
    select_subset,
)
from rag_engine.retrieval.retriever import RetrievedChunk, Retriever

__all__ = [
    "QUESTION_TEMPLATE",
    "AnswerEvalItem",
    "AnswerEvalRecord",
    "build_answer_report",
    "build_items",
    "format_question",
    "pick_subsets",
    "run_answer_eval",
    "write_answer_report",
]

QUESTION_TEMPLATE: str = (
    "Claim: {claim}\n"
    "Do the contexts support or contradict this claim? Answer briefly, citing the evidence. "
    "If the contexts do not address the claim, abstain."
)


def format_question(claim: str) -> str:
    return QUESTION_TEMPLATE.format(claim=claim)


@dataclass(frozen=True)
class AnswerEvalItem:
    query_id: str
    kind: ItemKind
    question: str
    hits: tuple[RetrievedChunk, ...]
    gold_doc_ids: tuple[str, ...]


def build_items(
    *,
    retriever: Retriever,
    claims: Mapping[str, str],
    gold: Mapping[str, Collection[str]],
    answerable_ids: Sequence[str],
    no_relevant_ids: Sequence[str],
    k: int,
    depth: int,
) -> list[AnswerEvalItem]:
    if k < 1:
        raise ValueError("k must be at least 1")
    if depth < k:
        raise ValueError("depth must be at least k")

    overlap = set(answerable_ids) & set(no_relevant_ids)
    if overlap:
        raise ValueError(f"ids appear in both subsets: {sorted(overlap)}")

    items: list[AnswerEvalItem] = []

    for query_id in answerable_ids:
        claim = claims[query_id]
        hits = retriever.retrieve(claim, k=depth)
        if len(hits) < k:
            raise ValueError(f"only {len(hits)} hits for answerable query {query_id!r}; need {k}")
        items.append(
            AnswerEvalItem(
                query_id=query_id,
                kind=ItemKind.ANSWERABLE,
                question=format_question(claim),
                hits=tuple(hits[:k]),
                gold_doc_ids=tuple(sorted(gold[query_id])),
            )
        )

    for query_id in no_relevant_ids:
        claim = claims[query_id]
        hits = retriever.retrieve(claim, k=depth)
        selected = drop_gold_hits(hits, gold[query_id], k=k)
        # Exactly k contexts for every item, so abstention is not made easier by short lists.
        if len(selected) < k:
            raise ValueError(
                f"only {len(selected)} non-gold hits remain for no-relevant query "
                f"{query_id!r}; need {k}"
            )
        items.append(
            AnswerEvalItem(
                query_id=query_id,
                kind=ItemKind.NO_RELEVANT,
                question=format_question(claim),
                hits=tuple(selected),
                gold_doc_ids=tuple(sorted(gold[query_id])),
            )
        )

    return items


def pick_subsets(
    labelled_ids: Iterable[str],
    *,
    n_answerable: int,
    n_no_relevant: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    ids = list(labelled_ids)
    if n_answerable > len(ids):
        raise ValueError("not enough ids for answerable subset")

    answerable = select_subset(ids, n_answerable, seed=seed)
    selected = set(answerable)
    remaining = [query_id for query_id in ids if query_id not in selected]

    if n_no_relevant > len(remaining):
        raise ValueError("not enough ids for no-relevant subset")

    no_relevant: list[str]
    if n_no_relevant == 0:
        no_relevant = []
    else:
        no_relevant = select_subset(remaining, n_no_relevant, seed=seed + 1)

    return answerable, no_relevant


class AnswerEvalRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    kind: ItemKind
    question: str
    contexts: tuple[dict[str, str | int], ...]
    gold_doc_ids: tuple[str, ...]
    raw_contents: tuple[str, ...]
    response: dict[str, Any] | None
    error: str | None
    grade: AnswerGrade


def _context_record(context: EvalContext) -> dict[str, str | int]:
    return {
        "index": context.index,
        "doc_id": context.doc_id,
        "chunk_id": context.chunk_id,
        "text_sha256": context.text_sha256,
    }


def run_answer_eval(
    items: Sequence[AnswerEvalItem],
    generator: CachedAnswerGenerator,
    *,
    max_sentences: int = 3,
    on_item: Callable[[int, AnswerEvalRecord], None] | None = None,
) -> list[AnswerEvalRecord]:
    records: list[AnswerEvalRecord] = []

    for index, item in enumerate(items):
        contexts = contexts_from_hits(item.hits)
        response: AnswerResponse | None = None
        error: str | None = None

        try:
            response = generator.generate(item.question, item.hits, max_sentences=max_sentences)
        except CacheMissError:
            raise
        except ValueError as exc:
            response = None
            error = str(exc)

        raw_content = generator.last_raw_content
        grade = grade_answer(
            query_id=item.query_id,
            kind=item.kind,
            contexts=contexts,
            gold_doc_ids=item.gold_doc_ids,
            raw_content=raw_content,
            response=response,
            error=error,
        )

        record = AnswerEvalRecord(
            query_id=item.query_id,
            kind=item.kind,
            question=item.question,
            contexts=tuple(_context_record(context) for context in contexts),
            gold_doc_ids=item.gold_doc_ids,
            raw_contents=tuple(generator.raw_contents),
            response=response.model_dump(mode="json") if response is not None else None,
            error=error,
            grade=grade,
        )
        records.append(record)
        if on_item is not None:
            on_item(index, record)

    return records


def build_answer_report(
    records: Sequence[AnswerEvalRecord],
    *,
    config: Mapping[str, Any],
    created_at: str,
    git_sha: str,
    git_dirty: bool | None,
    dataset_name: str,
    dataset_checksum: str,
    n_cache_hits: int,
    n_api_calls: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")

    return {
        "schema_version": ANSWER_EVAL_SCHEMA_VERSION,
        "kind": "answer_eval",
        "created_at": created_at,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "dataset_name": dataset_name,
        "dataset_checksum": dataset_checksum,
        "selection_method": SELECTION_METHOD,
        "question_template": QUESTION_TEMPLATE,
        "config": dict(config),
        "grading": (
            "code-only: citation span validity, citation recall/precision vs qrels gold docs, "
            "abstention; no LLM judge"
        ),
        "n_cache_hits": n_cache_hits,
        "n_api_calls": n_api_calls,
        "metrics": aggregate_grades([record.grade for record in records]),
        "records": [record.model_dump(mode="json") for record in records],
    }


def write_answer_report(report: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path
