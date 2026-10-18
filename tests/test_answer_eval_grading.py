"""Unit tests for rag_engine.eval.answers."""

from __future__ import annotations

import hashlib

import pytest

from rag_engine.answering.models import AnswerResponse, Citation
from rag_engine.eval import answers
from rag_engine.eval.answers import (
    ANSWER_EVAL_SCHEMA_VERSION,
    MIN_QUOTE_CHARS,
    SELECTION_METHOD,
    AnswerGrade,
    EvalContext,
    ItemKind,
    aggregate_grades,
    contexts_from_hits,
    drop_gold_hits,
    grade_answer,
    parse_raw_answer,
    select_subset,
)
from rag_engine.models import Chunk
from rag_engine.retrieval.retriever import RetrievedChunk


def make_chunk(
    doc_id: str,
    chunk_id: str,
    text: str,
    *,
    start_char: int = 0,
    index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        index=index,
        text=text,
        start_char=start_char,
        end_char=start_char + len(text),
        token_count=max(1, len(text.split())),
        heading_path=(),
        page=None,
        context_header="",
        metadata={},
    )


def make_hit(
    doc_id: str,
    chunk_id: str,
    text: str,
    *,
    start_char: int = 0,
    score: float = 1.0,
    rank: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=make_chunk(doc_id, chunk_id, text, start_char=start_char),
        score=score,
        rank=rank,
        retriever="test",
    )


def make_citation(
    *,
    doc_id: str,
    chunk_id: str,
    start_char: int,
    end_char: int,
    quote: str,
) -> Citation:
    return Citation(
        doc_id=doc_id,
        chunk_id=chunk_id,
        source="test",
        content_hash="hash",
        heading_path=(),
        page=None,
        page_end=None,
        start_char=start_char,
        end_char=end_char,
        quote=quote,
    )


def make_response(
    *,
    query: str = "q",
    answer: str = "a",
    citations: tuple[Citation, ...] = (),
    abstained: bool = False,
) -> AnswerResponse:
    return AnswerResponse(
        schema_version=1,
        query=query,
        answer=answer,
        citations=citations,
        generator="test",
        abstained=abstained,
        model_name=None,
    )


def make_grade(
    *,
    query_id: str = "q",
    kind: ItemKind = ItemKind.ANSWERABLE,
    parse_ok: bool = True,
    abstained: bool = False,
    n_raw_citations: int = 0,
    n_valid_citations: int = 0,
    n_valid_gold_citations: int = 0,
    final_citations_consistent: bool = True,
    gold_doc_ids_in_context: tuple[str, ...] = (),
    cited_doc_ids: tuple[str, ...] = (),
    citation_recall: float | None = None,
    citation_precision: float | None = None,
    abstention_correct: bool | None = None,
    error: str | None = None,
) -> AnswerGrade:
    return AnswerGrade(
        query_id=query_id,
        kind=kind,
        parse_ok=parse_ok,
        abstained=abstained,
        n_raw_citations=n_raw_citations,
        n_valid_citations=n_valid_citations,
        n_valid_gold_citations=n_valid_gold_citations,
        final_citations_consistent=final_citations_consistent,
        gold_doc_ids_in_context=gold_doc_ids_in_context,
        cited_doc_ids=cited_doc_ids,
        citation_recall=citation_recall,
        citation_precision=citation_precision,
        abstention_correct=abstention_correct,
        error=error,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants() -> None:
    assert ANSWER_EVAL_SCHEMA_VERSION == 1
    assert "random.Random" in SELECTION_METHOD


def test_min_quote_chars_exported() -> None:
    assert MIN_QUOTE_CHARS == 20
    assert "MIN_QUOTE_CHARS" in answers.__all__


# ---------------------------------------------------------------------------
# select_subset
# ---------------------------------------------------------------------------


def test_select_subset_deterministic_sorted_dedupe() -> None:
    query_ids = ["b", "a", "c", "a", "b", "d"]
    out1 = select_subset(query_ids, 3, seed=42)
    out2 = select_subset(query_ids, 3, seed=42)
    assert out1 == out2
    assert out1 == sorted(out1)
    assert len(out1) == 3
    assert set(out1) <= {"a", "b", "c", "d"}
    assert len(set(out1)) == 3


def test_select_subset_different_seeds_can_differ() -> None:
    pool = [f"q{i}" for i in range(50)]
    results = {tuple(select_subset(pool, 10, seed=s)) for s in range(5)}
    assert len(results) >= 2


@pytest.mark.parametrize("n", [0, -1, 4])
def test_select_subset_value_errors(n: int) -> None:
    with pytest.raises(ValueError, match="n must be between"):
        select_subset(["a", "b", "c"], n, seed=0)


# ---------------------------------------------------------------------------
# EvalContext / contexts_from_hits
# ---------------------------------------------------------------------------


def test_eval_context_text_sha256() -> None:
    ctx = EvalContext(index=0, doc_id="d", chunk_id="c", start_char=0, text="hello")
    expected = hashlib.sha256(b"hello").hexdigest()
    assert ctx.text_sha256 == expected


def test_contexts_from_hits() -> None:
    hits = [
        make_hit("d1", "c1", "hello", start_char=0),
        make_hit("d2", "c2", "world", start_char=10),
    ]
    contexts = contexts_from_hits(hits)
    assert [c.index for c in contexts] == [0, 1]
    assert contexts[0].doc_id == "d1"
    assert contexts[0].chunk_id == "c1"
    assert contexts[0].start_char == 0
    assert contexts[0].text == "hello"
    assert contexts[1].start_char == 10
    assert contexts[1].text_sha256 == hashlib.sha256(b"world").hexdigest()


# ---------------------------------------------------------------------------
# drop_gold_hits
# ---------------------------------------------------------------------------


def test_drop_gold_hits_k_error() -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        drop_gold_hits([], {"d"}, k=0)


def test_drop_gold_hits_order_and_limit() -> None:
    hits = [
        make_hit("d1", "c1", "a"),
        make_hit("gold", "c2", "b"),
        make_hit("d2", "c3", "c"),
        make_hit("d3", "c4", "d"),
    ]
    out = drop_gold_hits(hits, {"gold"}, k=2)
    assert [h.chunk.chunk_id for h in out] == ["c1", "c3"]
    # Fewer than k non-gold hits remain.
    out2 = drop_gold_hits(hits, {"gold"}, k=10)
    assert [h.chunk.chunk_id for h in out2] == ["c1", "c3", "c4"]


# ---------------------------------------------------------------------------
# parse_raw_answer
# ---------------------------------------------------------------------------


def test_parse_raw_answer_plain_json() -> None:
    raw = parse_raw_answer(
        '{"answer": "hi", "abstained": false, "citations": [{"chunk_index": 0, "quote": "a"}]}'
    )
    assert raw.parse_ok is True
    assert raw.answer == "hi"
    assert raw.abstained is False
    assert raw.citations == ((0, "a"),)
    assert raw.error is None


def test_parse_raw_answer_fenced_with_preamble() -> None:
    content = 'Sure thing!\n```json\n{"answer": "yes", "abstained": false}\n```\n'
    raw = parse_raw_answer(content)
    assert raw.parse_ok is True
    assert raw.answer == "yes"


def test_parse_raw_answer_fence_no_brace() -> None:
    raw = parse_raw_answer("```\nnot json\n```")
    assert raw.parse_ok is False
    assert raw.error is not None
    assert raw.error.startswith("invalid JSON")


def test_parse_raw_answer_fence_no_close() -> None:
    raw = parse_raw_answer('```json\n{"answer": "x", "abstained": false}')
    assert raw.parse_ok is True
    assert raw.answer == "x"


def test_parse_raw_answer_invalid_json() -> None:
    raw = parse_raw_answer("not json")
    assert raw.parse_ok is False
    assert raw.error is not None
    assert raw.error.startswith("invalid JSON")


def test_parse_raw_answer_non_object() -> None:
    raw = parse_raw_answer("[1, 2, 3]")
    assert raw.parse_ok is False
    assert raw.error == "model output must be a JSON object"


def test_parse_raw_answer_bad_answer_type() -> None:
    raw = parse_raw_answer('{"answer": 5}')
    assert raw.parse_ok is False
    assert raw.error == "answer must be a string"


def test_parse_raw_answer_bad_abstained_type() -> None:
    raw = parse_raw_answer('{"answer": "a", "abstained": "yes"}')
    assert raw.parse_ok is False
    assert raw.error == "abstained must be a boolean"


def test_parse_raw_answer_bad_citations_type() -> None:
    raw = parse_raw_answer('{"answer": "a", "citations": {}}')
    assert raw.parse_ok is False
    assert raw.error == "citations must be a list"


def test_parse_raw_answer_citation_entry_not_dict() -> None:
    raw = parse_raw_answer('{"answer": "a", "citations": ["nope"]}')
    assert raw.parse_ok is True
    assert raw.citations == ((None, None),)


def test_parse_raw_answer_bool_chunk_index() -> None:
    raw = parse_raw_answer('{"answer": "a", "citations": [{"chunk_index": true, "quote": "q"}]}')
    assert raw.parse_ok is True
    assert raw.citations == ((None, "q"),)


def test_parse_raw_answer_non_str_quote() -> None:
    raw = parse_raw_answer('{"answer": "a", "citations": [{"chunk_index": 0, "quote": 7}]}')
    assert raw.parse_ok is True
    assert raw.citations == ((0, None),)


def test_parse_raw_answer_empty_answer_forces_abstained() -> None:
    raw = parse_raw_answer('{"answer": "   ", "abstained": false}')
    assert raw.parse_ok is True
    assert raw.abstained is True


def test_parse_raw_answer_abstained_true() -> None:
    raw = parse_raw_answer('{"answer": "something", "abstained": true}')
    assert raw.parse_ok is True
    assert raw.abstained is True


# ---------------------------------------------------------------------------
# grade_answer
# ---------------------------------------------------------------------------


def test_grade_answer_valid_citation_and_gold() -> None:
    quote = "hello world from the first context"
    contexts = [
        EvalContext(index=0, doc_id="A", chunk_id="c0", start_char=0, text=quote),
        EvalContext(
            index=1,
            doc_id="B",
            chunk_id="c1",
            start_char=0,
            text="a second context for document B",
        ),
    ]
    raw = (
        '{"answer": "hi", "abstained": false, "citations": ['
        f'{{"chunk_index": 0, "quote": "{quote}"}}]}}'
    )
    resp = make_response(
        answer="hi",
        citations=(
            make_citation(
                doc_id="A",
                chunk_id="c0",
                start_char=0,
                end_char=len(quote),
                quote=quote,
            ),
        ),
    )
    grade = grade_answer(
        query_id="q1",
        kind=ItemKind.ANSWERABLE,
        contexts=contexts,
        gold_doc_ids={"A"},
        raw_content=raw,
        response=resp,
    )
    assert grade.query_id == "q1"
    assert grade.kind is ItemKind.ANSWERABLE
    assert grade.parse_ok is True
    assert grade.abstained is False
    assert grade.n_raw_citations == 1
    assert grade.n_valid_citations == 1
    assert grade.n_valid_gold_citations == 1
    assert grade.final_citations_consistent is True
    assert grade.gold_doc_ids_in_context == ("A",)
    assert grade.cited_doc_ids == ("A",)
    assert grade.citation_recall == pytest.approx(1.0)
    assert grade.citation_precision == pytest.approx(1.0)
    assert grade.abstention_correct is True
    assert grade.error is None


def test_grade_answer_invalid_citations() -> None:
    contexts = [
        EvalContext(
            index=0,
            doc_id="A",
            chunk_id="c0",
            start_char=0,
            text="hello world from the first context",
        ),
    ]
    raw = (
        '{"answer": "x", "abstained": false, "citations": ['
        '{"chunk_index": 5, "quote": "hello world from the first context"},'
        '{"chunk_index": 0, "quote": "text that is not in the context"},'
        '{"chunk_index": 0, "quote": "   "},'
        '{"chunk_index": 0, "quote": "HELLO WORLD FROM THE FIRST CONTEXT"}'
        "]}"
    )
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=contexts,
        gold_doc_ids={"A"},
        raw_content=raw,
        response=None,
    )
    assert grade.n_raw_citations == 4
    assert grade.n_valid_citations == 0
    assert grade.n_valid_gold_citations == 0
    assert grade.cited_doc_ids == ()
    assert grade.citation_recall == pytest.approx(0.0)
    assert grade.citation_precision is None


def test_grade_answer_nineteen_char_quote_is_invalid() -> None:
    # A 19-character quote is a verbatim substring but below the minimum length.
    quote = "hello world from th"
    assert len(quote) == MIN_QUOTE_CHARS - 1
    ctx = EvalContext(
        index=0,
        doc_id="A",
        chunk_id="c0",
        start_char=0,
        text="hello world from the first context",
    )
    assert quote in ctx.text
    raw = f'{{"answer": "x", "citations": [{{"chunk_index": 0, "quote": "{quote}"}}]}}'
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[ctx],
        gold_doc_ids={"A"},
        raw_content=raw,
        response=None,
    )
    assert grade.n_raw_citations == 1
    assert grade.n_valid_citations == 0
    assert grade.cited_doc_ids == ()
    assert grade.citation_recall == pytest.approx(0.0)
    assert grade.citation_precision is None


def test_grade_answer_twenty_char_quote_is_valid() -> None:
    # Exactly MIN_QUOTE_CHARS characters is enough when the span is grounded.
    quote = "hello world from the"
    assert len(quote) == MIN_QUOTE_CHARS
    ctx = EvalContext(
        index=0,
        doc_id="A",
        chunk_id="c0",
        start_char=0,
        text="hello world from the first context",
    )
    assert quote in ctx.text
    raw = f'{{"answer": "x", "citations": [{{"chunk_index": 0, "quote": "{quote}"}}]}}'
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[ctx],
        gold_doc_ids={"A"},
        raw_content=raw,
        response=None,
    )
    assert grade.n_raw_citations == 1
    assert grade.n_valid_citations == 1
    assert grade.n_valid_gold_citations == 1
    assert grade.cited_doc_ids == ("A",)
    assert grade.citation_precision == pytest.approx(1.0)


def test_grade_answer_stripped_quote_below_minimum_is_invalid() -> None:
    # The raw quote is 21 characters and appears verbatim in the context, but its
    # stripped form is only 19 characters, so the span is not credible evidence.
    context_text = "  hello world from th and then some more text"
    quote = "  hello world from th"
    assert len(quote) >= MIN_QUOTE_CHARS
    assert len(quote.strip()) < MIN_QUOTE_CHARS
    assert quote in context_text
    ctx = EvalContext(index=0, doc_id="A", chunk_id="c0", start_char=0, text=context_text)
    raw = f'{{"answer": "x", "citations": [{{"chunk_index": 0, "quote": "{quote}"}}]}}'
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[ctx],
        gold_doc_ids={"A"},
        raw_content=raw,
        response=None,
    )
    assert grade.n_raw_citations == 1
    assert grade.n_valid_citations == 0
    assert grade.cited_doc_ids == ()
    assert grade.citation_recall == pytest.approx(0.0)
    assert grade.citation_precision is None


def test_grade_answer_recall_precision_two_gold() -> None:
    contexts = [
        EvalContext(
            index=0,
            doc_id="A",
            chunk_id="c0",
            start_char=0,
            text="alpha is the first context text",
        ),
        EvalContext(
            index=1,
            doc_id="B",
            chunk_id="c1",
            start_char=0,
            text="beta is the second context text",
        ),
        EvalContext(
            index=2,
            doc_id="C",
            chunk_id="c2",
            start_char=0,
            text="gamma is the third context text",
        ),
    ]
    raw = (
        '{"answer": "x", "citations": ['
        '{"chunk_index": 0, "quote": "alpha is the first context text"},'
        '{"chunk_index": 2, "quote": "gamma is the third context text"}'
        "]}"
    )
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=contexts,
        gold_doc_ids={"A", "B"},
        raw_content=raw,
        response=None,
    )
    assert grade.cited_doc_ids == ("A", "C")
    assert grade.n_valid_citations == 2
    assert grade.n_valid_gold_citations == 1
    # 1 of 2 gold docs cited.
    assert grade.citation_recall == pytest.approx(0.5)
    # 1 of 2 valid citations is gold.
    assert grade.citation_precision == pytest.approx(0.5)


def test_grade_answer_abstained_answerable() -> None:
    contexts = [EvalContext(index=0, doc_id="A", chunk_id="c0", start_char=0, text="hello")]
    raw = '{"answer": "", "abstained": true}'
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=contexts,
        gold_doc_ids={"A"},
        raw_content=raw,
        response=None,
    )
    assert grade.abstained is True
    assert grade.citation_recall == pytest.approx(0.0)
    assert grade.abstention_correct is False


def test_grade_answer_no_relevant_abstained() -> None:
    contexts = [EvalContext(index=0, doc_id="X", chunk_id="c0", start_char=0, text="hello")]
    raw = '{"answer": "", "abstained": true}'
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.NO_RELEVANT,
        contexts=contexts,
        gold_doc_ids=set(),
        raw_content=raw,
        response=None,
    )
    assert grade.abstention_correct is True
    assert grade.citation_recall is None
    assert grade.citation_precision is None


def test_grade_answer_no_relevant_unparseable() -> None:
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.NO_RELEVANT,
        contexts=[],
        gold_doc_ids=set(),
        raw_content="not json",
        response=None,
    )
    assert grade.parse_ok is False
    assert grade.abstained is False
    assert grade.abstention_correct is False
    assert grade.error is not None
    assert grade.error.startswith("invalid JSON")


def test_grade_answer_answerable_no_gold_in_context() -> None:
    contexts = [EvalContext(index=0, doc_id="X", chunk_id="c0", start_char=0, text="hello")]
    raw = '{"answer": "x", "abstained": false}'
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=contexts,
        gold_doc_ids={"Z"},
        raw_content=raw,
        response=None,
    )
    assert grade.gold_doc_ids_in_context == ()
    assert grade.citation_recall is None
    assert grade.abstention_correct is None


def test_grade_answer_no_model_output() -> None:
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[],
        gold_doc_ids=set(),
        raw_content=None,
        response=None,
    )
    assert grade.parse_ok is False
    assert grade.error == "no model output"
    assert grade.n_raw_citations == 0


def test_grade_answer_no_model_output_custom_error() -> None:
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[],
        gold_doc_ids=set(),
        raw_content=None,
        response=None,
        error="boom",
    )
    assert grade.error == "boom"


def test_grade_answer_parse_ok_but_error_passed() -> None:
    raw = '{"answer": "x", "abstained": false}'
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[],
        gold_doc_ids=set(),
        raw_content=raw,
        response=None,
        error="leftover",
    )
    assert grade.parse_ok is True
    assert grade.error == "leftover"


def test_final_citations_consistent_none_response() -> None:
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[],
        gold_doc_ids=set(),
        raw_content='{"answer": "x"}',
        response=None,
    )
    assert grade.final_citations_consistent is None


def test_final_citations_consistent_no_citations() -> None:
    resp = make_response(citations=())
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[],
        gold_doc_ids=set(),
        raw_content='{"answer": "x"}',
        response=resp,
    )
    assert grade.final_citations_consistent is None


def test_final_citations_consistent_unknown_chunk_id() -> None:
    ctx = EvalContext(index=0, doc_id="A", chunk_id="c0", start_char=0, text="hello")
    resp = make_response(
        citations=(
            make_citation(doc_id="A", chunk_id="unknown", start_char=0, end_char=5, quote="hello"),
        )
    )
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[ctx],
        gold_doc_ids=set(),
        raw_content='{"answer": "x"}',
        response=resp,
    )
    assert grade.final_citations_consistent is False


def test_final_citations_consistent_wrong_offsets() -> None:
    ctx = EvalContext(index=0, doc_id="A", chunk_id="c0", start_char=100, text="hello")

    # Quote does not match the slice.
    resp = make_response(
        citations=(
            make_citation(doc_id="A", chunk_id="c0", start_char=100, end_char=105, quote="world"),
        )
    )
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[ctx],
        gold_doc_ids=set(),
        raw_content='{"answer": "x"}',
        response=resp,
    )
    assert grade.final_citations_consistent is False

    # local_start < 0 because citation starts before the context.
    resp = make_response(
        citations=(
            make_citation(doc_id="A", chunk_id="c0", start_char=99, end_char=105, quote="hello"),
        )
    )
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[ctx],
        gold_doc_ids=set(),
        raw_content='{"answer": "x"}',
        response=resp,
    )
    assert grade.final_citations_consistent is False

    # local_end > len(text).
    resp = make_response(
        citations=(
            make_citation(doc_id="A", chunk_id="c0", start_char=100, end_char=110, quote="hello"),
        )
    )
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[ctx],
        gold_doc_ids=set(),
        raw_content='{"answer": "x"}',
        response=resp,
    )
    assert grade.final_citations_consistent is False


def test_final_citations_consistent_doc_level_offsets() -> None:
    ctx = EvalContext(index=0, doc_id="A", chunk_id="c0", start_char=100, text="hello world")
    resp = make_response(
        citations=(
            make_citation(doc_id="A", chunk_id="c0", start_char=100, end_char=105, quote="hello"),
        )
    )
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[ctx],
        gold_doc_ids=set(),
        raw_content='{"answer": "x"}',
        response=resp,
    )
    assert grade.final_citations_consistent is True


# ---------------------------------------------------------------------------
# aggregate_grades
# ---------------------------------------------------------------------------


def test_aggregate_grades_all_keys() -> None:
    grades = [
        make_grade(
            query_id="q1",
            kind=ItemKind.ANSWERABLE,
            parse_ok=True,
            abstained=False,
            n_raw_citations=2,
            n_valid_citations=2,
            n_valid_gold_citations=1,
            final_citations_consistent=True,
            gold_doc_ids_in_context=("A",),
            cited_doc_ids=("A",),
            citation_recall=1.0,
            citation_precision=0.5,
            abstention_correct=True,
        ),
        make_grade(
            query_id="q2",
            kind=ItemKind.ANSWERABLE,
            parse_ok=True,
            abstained=True,
            n_raw_citations=1,
            n_valid_citations=0,
            n_valid_gold_citations=0,
            final_citations_consistent=True,
            gold_doc_ids_in_context=("B",),
            cited_doc_ids=(),
            citation_recall=0.0,
            citation_precision=None,
            abstention_correct=False,
        ),
        make_grade(
            query_id="q3",
            kind=ItemKind.NO_RELEVANT,
            parse_ok=True,
            abstained=True,
            n_raw_citations=0,
            n_valid_citations=0,
            n_valid_gold_citations=0,
            final_citations_consistent=True,
            abstention_correct=True,
        ),
        make_grade(
            query_id="q4",
            kind=ItemKind.NO_RELEVANT,
            parse_ok=False,
            abstained=False,
            final_citations_consistent=False,
            abstention_correct=False,
        ),
    ]
    agg = aggregate_grades(grades)
    assert agg["n_items"] == 4
    assert agg["n_answerable"] == 2
    assert agg["n_no_relevant"] == 2
    assert agg["n_parse_ok"] == 3
    assert agg["parse_ok_rate"] == pytest.approx(0.75)  # 3 / 4
    assert agg["n_raw_citations"] == 3  # 2 + 1 + 0 + 0
    assert agg["n_valid_citations"] == 2  # 2 + 0 + 0 + 0
    assert agg["citation_span_validity"] == pytest.approx(2 / 3)  # 2 / 3
    assert agg["n_recall_eligible"] == 2  # q1, q2
    assert agg["citation_recall"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2
    assert agg["citation_precision"] == pytest.approx(0.5)  # 1 / 2
    assert agg["n_answerable_gold_missing"] == 0
    assert agg["false_abstention_rate"] == pytest.approx(0.5)  # 1 / 2
    assert agg["correct_abstention_rate"] == pytest.approx(0.5)  # 1 / 2
    assert agg["final_citations_consistent_rate"] == pytest.approx(0.75)  # 3 / 4


def test_aggregate_grades_zero_denominators() -> None:
    grades = [
        make_grade(
            query_id="q",
            kind=ItemKind.ANSWERABLE,
            parse_ok=True,
            abstained=False,
            gold_doc_ids_in_context=(),
            citation_recall=None,
            citation_precision=None,
            abstention_correct=None,
        ),
    ]
    agg = aggregate_grades(grades)
    assert agg["n_items"] == 1
    assert agg["n_no_relevant"] == 0
    assert agg["n_raw_citations"] == 0
    assert agg["citation_span_validity"] is None
    assert agg["n_recall_eligible"] == 0
    assert agg["citation_recall"] is None
    assert agg["citation_precision"] is None
    assert agg["n_answerable_gold_missing"] == 1
    assert agg["false_abstention_rate"] is None
    assert agg["correct_abstention_rate"] is None
    assert agg["final_citations_consistent_rate"] == pytest.approx(1.0)


def test_aggregate_grades_empty() -> None:
    with pytest.raises(ValueError, match="grades must not be empty"):
        aggregate_grades([])


def test_raw_citation_with_missing_index_or_quote_is_invalid() -> None:
    # Both entries parse to (None, "Cats napping in the afternoon") and (0, None):
    # neither can be checked, even though the quote is long enough to pass alone.
    ctx = EvalContext(
        index=0,
        doc_id="A",
        chunk_id="c0",
        start_char=0,
        text="Cats napping in the afternoon sun.",
    )
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[ctx],
        gold_doc_ids={"A"},
        raw_content=(
            '{"answer": "x", "citations": ['
            '{"quote": "Cats napping in the afternoon"},'
            '{"chunk_index": 0}'
            "]}"
        ),
        response=None,
    )
    assert grade.n_raw_citations == 2
    assert grade.n_valid_citations == 0
    assert grade.citation_recall == 0.0
    assert grade.citation_precision is None


def test_unparseable_reply_is_never_a_correct_answer() -> None:
    # Gold doc A is in context, but the reply is not JSON: not "correctly answered".
    ctx = EvalContext(index=0, doc_id="A", chunk_id="c0", start_char=0, text="Cats nap.")
    grade = grade_answer(
        query_id="q",
        kind=ItemKind.ANSWERABLE,
        contexts=[ctx],
        gold_doc_ids={"A"},
        raw_content="not json",
        response=None,
    )
    assert grade.parse_ok is False
    assert grade.abstention_correct is False
    assert grade.final_citations_consistent is None
    agg = aggregate_grades([grade])
    assert agg["final_citations_consistent_rate"] is None
    assert agg["n_final_citations_checked"] == 0
