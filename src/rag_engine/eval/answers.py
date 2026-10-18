"""Offline, code-first grading for the M5 answer-generation evaluation slice.

Grades are computed from both the raw model reply and the final
``AnswerResponse``. The OpenAI-compatible generator silently drops citations
whose quote cannot be found in the cited chunk, so looking only at the final
response would hide invalid spans. No LLM judge is used here; every decision is
deterministic Python code.

A raw citation quote must also be at least :data:`MIN_QUOTE_CHARS` characters
long after stripping, because a very short quote gives no evidence that the
model grounded a real span.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Collection, Iterable, Sequence
from enum import StrEnum
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.answering.models import AnswerResponse
from rag_engine.retrieval.retriever import RetrievedChunk

__all__: list[str] = [
    "ANSWER_EVAL_SCHEMA_VERSION",
    "MIN_QUOTE_CHARS",
    "SELECTION_METHOD",
    "AnswerGrade",
    "EvalContext",
    "ItemKind",
    "RawAnswer",
    "aggregate_grades",
    "contexts_from_hits",
    "drop_gold_hits",
    "grade_answer",
    "parse_raw_answer",
    "select_subset",
]

ANSWER_EVAL_SCHEMA_VERSION: int = 1
MIN_QUOTE_CHARS: int = 20
SELECTION_METHOD: str = "random.Random(seed).sample(sorted(set(query_ids)), n), returned sorted"


class ItemKind(StrEnum):
    """Whether the query should be answered or refused."""

    ANSWERABLE = "answerable"
    NO_RELEVANT = "no_relevant"


def select_subset(query_ids: Iterable[str], n: int, *, seed: int) -> list[str]:
    """Select a deterministic random subset of query ids."""

    pool = sorted(set(query_ids))
    if n < 1 or n > len(pool):
        raise ValueError(f"n must be between 1 and {len(pool)}, got {n}")
    rng = random.Random(seed)
    return sorted(rng.sample(pool, n))


class EvalContext(BaseModel):
    """One retrieved context presented to the answer generator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    doc_id: str
    chunk_id: str
    start_char: int = Field(ge=0)
    text: str

    @property
    def text_sha256(self) -> str:
        """Hex SHA-256 digest of the context text."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def contexts_from_hits(hits: Sequence[RetrievedChunk]) -> list[EvalContext]:
    """Convert retrieved chunks into indexed evaluation contexts."""

    return [
        EvalContext(
            index=index,
            doc_id=hit.chunk.doc_id,
            chunk_id=hit.chunk.chunk_id,
            start_char=hit.chunk.start_char,
            text=hit.chunk.text,
        )
        for index, hit in enumerate(hits)
    ]


def drop_gold_hits(
    hits: Sequence[RetrievedChunk],
    gold_doc_ids: Collection[str],
    *,
    k: int,
) -> list[RetrievedChunk]:
    """Remove gold documents and keep the first ``k`` remaining hits."""

    if k < 1:
        raise ValueError("k must be at least 1")
    gold = set(gold_doc_ids)
    kept: list[RetrievedChunk] = []
    for hit in hits:
        if hit.chunk.doc_id in gold:
            continue
        kept.append(hit)
        if len(kept) == k:
            break
    return kept


class RawAnswer(BaseModel):
    """Parsed representation of the model's raw JSON reply."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parse_ok: bool
    answer: str = ""
    abstained: bool = False
    citations: tuple[tuple[int | None, str | None], ...] = ()
    error: str | None = None


def _extract_json_payload(content: str) -> str:
    """Return the first fenced block if present, else the stripped content."""

    stripped = content.strip()
    if "```" not in stripped:
        return stripped
    fence_start = stripped.find("```")
    remainder = stripped[fence_start + 3 :]
    brace = remainder.find("{")
    body = remainder if brace == -1 else remainder[brace:]
    fence_end = body.find("```")
    if fence_end != -1:
        body = body[:fence_end]
    return body.strip()


def parse_raw_answer(content: str) -> RawAnswer:
    """Parse a raw model reply without ever raising on malformed output."""

    try:
        payload = _extract_json_payload(content)
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        return RawAnswer(parse_ok=False, error=f"invalid JSON: {exc}")

    if not isinstance(loaded, dict):
        return RawAnswer(parse_ok=False, error="model output must be a JSON object")

    data = cast("dict[str, object]", loaded)

    answer_value = data.get("answer", "")
    if not isinstance(answer_value, str):
        return RawAnswer(parse_ok=False, error="answer must be a string")
    answer = answer_value

    abstained_value = data.get("abstained", False)
    if not isinstance(abstained_value, bool):
        return RawAnswer(parse_ok=False, error="abstained must be a boolean")
    abstained = abstained_value

    citations_value = data.get("citations", [])
    if not isinstance(citations_value, list):
        return RawAnswer(parse_ok=False, error="citations must be a list")

    citations: list[tuple[int | None, str | None]] = []
    for entry in citations_value:
        if not isinstance(entry, dict):
            citations.append((None, None))
            continue
        entry_dict = cast("dict[str, object]", entry)
        chunk_index_value = entry_dict.get("chunk_index")
        quote_value = entry_dict.get("quote")
        chunk_index: int | None = None
        if isinstance(chunk_index_value, int) and not isinstance(chunk_index_value, bool):
            chunk_index = chunk_index_value
        quote: str | None = None
        if isinstance(quote_value, str):
            quote = quote_value
        citations.append((chunk_index, quote))

    if abstained or not answer.strip():
        abstained = True

    return RawAnswer(
        parse_ok=True,
        answer=answer,
        abstained=abstained,
        citations=tuple(citations),
        error=None,
    )


class AnswerGrade(BaseModel):
    """Deterministic grade for one answer-eval item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    kind: ItemKind
    parse_ok: bool
    abstained: bool
    n_raw_citations: int = Field(ge=0)
    n_valid_citations: int = Field(ge=0)
    n_valid_gold_citations: int = Field(ge=0)
    final_citations_consistent: bool | None
    gold_doc_ids_in_context: tuple[str, ...]
    cited_doc_ids: tuple[str, ...]
    citation_recall: float | None
    citation_precision: float | None
    abstention_correct: bool | None
    error: str | None = None


def _final_citations_consistent(
    contexts: Sequence[EvalContext],
    response: AnswerResponse | None,
) -> bool | None:
    """Whether every final citation is grounded in its context; None if there are none."""

    if response is None or not response.citations:
        return None
    by_chunk_id = {context.chunk_id: context for context in contexts}
    for citation in response.citations:
        context = by_chunk_id.get(citation.chunk_id)
        if context is None:
            return False
        local_start = citation.start_char - context.start_char
        local_end = citation.end_char - context.start_char
        # Citation itself guarantees end_char > start_char.
        if local_start < 0 or local_end > len(context.text):
            return False
        if context.text[local_start:local_end] != citation.quote:
            return False
    return True


def grade_answer(
    *,
    query_id: str,
    kind: ItemKind,
    contexts: Sequence[EvalContext],
    gold_doc_ids: Collection[str],
    raw_content: str | None,
    response: AnswerResponse | None,
    error: str | None = None,
) -> AnswerGrade:
    """Grade one answer-eval item from raw model content and final response."""

    gold = set(gold_doc_ids)

    final_error: str | None
    if raw_content is None:
        parse_ok = False
        abstained = False
        raw_citations: tuple[tuple[int | None, str | None], ...] = ()
        final_error = error or "no model output"
    else:
        raw = parse_raw_answer(raw_content)
        parse_ok = raw.parse_ok
        abstained = raw.abstained if parse_ok else False
        raw_citations = raw.citations if parse_ok else ()
        final_error = raw.error if not parse_ok else error

    valid_citations: list[EvalContext] = []
    for chunk_index, quote in raw_citations:
        if chunk_index is None or quote is None:
            continue
        if not (0 <= chunk_index < len(contexts)):
            continue
        context = contexts[chunk_index]
        if not quote.strip():
            continue
        # Short strings such as "the" or "BDNF" show up in a context by chance, so a
        # match on them is not evidence that the model grounded a real span.
        if len(quote.strip()) < MIN_QUOTE_CHARS:
            continue
        if quote not in context.text:
            continue
        valid_citations.append(context)

    n_raw_citations = len(raw_citations)
    n_valid_citations = len(valid_citations)
    n_valid_gold_citations = sum(1 for context in valid_citations if context.doc_id in gold)
    cited_doc_ids = tuple(sorted({context.doc_id for context in valid_citations}))
    gold_doc_ids_in_context = tuple(
        sorted({context.doc_id for context in contexts if context.doc_id in gold})
    )

    if not gold_doc_ids_in_context:
        citation_recall: float | None = None
    else:
        cited = set(cited_doc_ids)
        citation_recall = len(cited & set(gold_doc_ids_in_context)) / len(gold_doc_ids_in_context)

    if n_valid_citations == 0:
        citation_precision: float | None = None
    else:
        citation_precision = n_valid_gold_citations / n_valid_citations

    # An unparseable reply is neither a correct abstention nor a correct answer.
    if kind is ItemKind.NO_RELEVANT:
        abstention_correct: bool | None = parse_ok and abstained
    elif gold_doc_ids_in_context:
        abstention_correct = parse_ok and not abstained
    else:
        abstention_correct = None

    return AnswerGrade(
        query_id=query_id,
        kind=kind,
        parse_ok=parse_ok,
        abstained=abstained,
        n_raw_citations=n_raw_citations,
        n_valid_citations=n_valid_citations,
        n_valid_gold_citations=n_valid_gold_citations,
        final_citations_consistent=_final_citations_consistent(contexts, response),
        gold_doc_ids_in_context=gold_doc_ids_in_context,
        cited_doc_ids=cited_doc_ids,
        citation_recall=citation_recall,
        citation_precision=citation_precision,
        abstention_correct=abstention_correct,
        error=final_error,
    )


def aggregate_grades(
    grades: Sequence[AnswerGrade],
) -> dict[str, float | int | None]:
    """Aggregate per-item grades into reportable micro/macro metrics."""

    if not grades:
        raise ValueError("grades must not be empty")

    n_items = len(grades)
    n_answerable = sum(1 for grade in grades if grade.kind is ItemKind.ANSWERABLE)
    n_no_relevant = sum(1 for grade in grades if grade.kind is ItemKind.NO_RELEVANT)
    n_parse_ok = sum(1 for grade in grades if grade.parse_ok)
    n_raw_citations = sum(grade.n_raw_citations for grade in grades)
    n_valid_citations = sum(grade.n_valid_citations for grade in grades)
    n_valid_gold_citations = sum(grade.n_valid_gold_citations for grade in grades)

    recall_values = [grade.citation_recall for grade in grades if grade.citation_recall is not None]
    n_recall_eligible = len(recall_values)
    citation_recall = sum(recall_values) / n_recall_eligible if n_recall_eligible else None

    citation_span_validity = n_valid_citations / n_raw_citations if n_raw_citations else None
    citation_precision = n_valid_gold_citations / n_valid_citations if n_valid_citations else None

    n_answerable_gold_missing = sum(
        1
        for grade in grades
        if grade.kind is ItemKind.ANSWERABLE and not grade.gold_doc_ids_in_context
    )

    false_abstention_denominator = sum(
        1 for grade in grades if grade.kind is ItemKind.ANSWERABLE and grade.gold_doc_ids_in_context
    )
    false_abstention_numerator = sum(
        1
        for grade in grades
        if grade.kind is ItemKind.ANSWERABLE and grade.gold_doc_ids_in_context and grade.abstained
    )
    false_abstention_rate = (
        false_abstention_numerator / false_abstention_denominator
        if false_abstention_denominator
        else None
    )

    n_correct_abstention = sum(
        1 for grade in grades if grade.kind is ItemKind.NO_RELEVANT and grade.abstained
    )
    correct_abstention_rate = n_correct_abstention / n_no_relevant if n_no_relevant else None

    consistency = [
        grade.final_citations_consistent
        for grade in grades
        if grade.final_citations_consistent is not None
    ]
    final_citations_consistent_rate = (
        sum(1 for ok in consistency if ok) / len(consistency) if consistency else None
    )

    return {
        "n_items": n_items,
        "n_answerable": n_answerable,
        "n_no_relevant": n_no_relevant,
        "n_parse_ok": n_parse_ok,
        "parse_ok_rate": n_parse_ok / n_items,
        "n_raw_citations": n_raw_citations,
        "n_valid_citations": n_valid_citations,
        "citation_span_validity": citation_span_validity,
        "n_recall_eligible": n_recall_eligible,
        "citation_recall": citation_recall,
        "citation_precision": citation_precision,
        "n_answerable_gold_missing": n_answerable_gold_missing,
        "false_abstention_rate": false_abstention_rate,
        "correct_abstention_rate": correct_abstention_rate,
        "n_final_citations_checked": len(consistency),
        "final_citations_consistent_rate": final_citations_consistent_rate,
    }
