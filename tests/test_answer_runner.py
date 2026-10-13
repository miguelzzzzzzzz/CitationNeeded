"""Tests for rag_engine.eval.answer_runner."""

from __future__ import annotations

import email.message
import hashlib
import json
import urllib.error
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from rag_engine.eval.answer_cache import (
    CachedAnswerGenerator,
    CacheMissError,
    ResponseCache,
)
from rag_engine.eval.answer_runner import (
    QUESTION_TEMPLATE,
    AnswerEvalItem,
    AnswerEvalRecord,
    build_answer_report,
    build_items,
    format_question,
    pick_subsets,
    run_answer_eval,
    write_answer_report,
)
from rag_engine.eval.answers import (
    ANSWER_EVAL_SCHEMA_VERSION,
    SELECTION_METHOD,
    ItemKind,
)
from rag_engine.models import Chunk
from rag_engine.retrieval.retriever import RetrievedChunk

BASE_URL = "https://api.example.test/v1"
MODEL = "m"
TEXT0 = "Aspirin reduces fever quickly."
TEXT1 = "Vitamin C does not cure colds."


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


def make_chunk(chunk_id: str, doc_id: str, index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        index=index,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
        heading_path=(),
        page=None,
        metadata={"source": f"{doc_id}.txt"},
    )


def make_hit(chunk: Chunk, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=1.0, rank=rank, retriever="fake")


class FakeRetriever:
    name = "fake"

    def __init__(self, responses: Mapping[str, Sequence[RetrievedChunk]] | None = None) -> None:
        self._responses = {q: tuple(v) for q, v in (responses or {}).items()}
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return list(self._responses.get(query, ()))[:k]


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeHTTP:
    def __init__(self, responses: list[bytes | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[Any] = []

    def __call__(self, req: Any, timeout: float | None = None) -> Any:
        self.calls.append(req)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


def envelope(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


def make_generator(tmp_path: Path, *, replay_only: bool = False) -> CachedAnswerGenerator:
    return CachedAnswerGenerator(
        cache=ResponseCache(tmp_path / "cache.jsonl"),
        replay_only=replay_only,
        base_url=BASE_URL,
        model=MODEL,
    )


def make_answerable_item() -> AnswerEvalItem:
    chunk = make_chunk("c0", "d1", 0, TEXT0)
    return AnswerEvalItem(
        query_id="q0",
        kind=ItemKind.ANSWERABLE,
        question=format_question("Aspirin reduces fever."),
        hits=(make_hit(chunk),),
        gold_doc_ids=("d1",),
    )


def make_no_relevant_item() -> AnswerEvalItem:
    chunk = make_chunk("c1", "d2", 0, TEXT1)
    return AnswerEvalItem(
        query_id="q1",
        kind=ItemKind.NO_RELEVANT,
        question=format_question("Vitamin C cures colds."),
        hits=(make_hit(chunk),),
        gold_doc_ids=("d9",),
    )


def answerable_content() -> str:
    return json.dumps(
        {
            "answer": "Yes.",
            "citations": [{"chunk_index": 0, "quote": TEXT0}],
            "abstained": False,
        }
    )


def abstain_content() -> str:
    return json.dumps({"answer": "Cannot answer.", "citations": [], "abstained": True})


def _run_basic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[AnswerEvalRecord]:
    items = [make_answerable_item(), make_no_relevant_item()]
    fake = FakeHTTP([envelope(answerable_content()), envelope(abstain_content())])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    return run_answer_eval(items, make_generator(tmp_path))


# ---------------------------------------------------------------------------
# format_question
# ---------------------------------------------------------------------------


def test_format_question_contains_claim() -> None:
    result = format_question("Aspirin reduces fever.")
    assert "Aspirin reduces fever." in result
    assert result == QUESTION_TEMPLATE.format(claim="Aspirin reduces fever.")


# ---------------------------------------------------------------------------
# pick_subsets
# ---------------------------------------------------------------------------


def test_pick_subsets_disjoint_and_deterministic() -> None:
    ids = [f"q{i}" for i in range(10)]
    a1, n1 = pick_subsets(ids, n_answerable=3, n_no_relevant=2, seed=42)
    a2, n2 = pick_subsets(ids, n_answerable=3, n_no_relevant=2, seed=42)
    assert a1 == a2
    assert n1 == n2
    assert len(a1) == 3
    assert len(n1) == 2
    assert set(a1).isdisjoint(n1)


def test_pick_subsets_zero_no_relevant() -> None:
    ids = [f"q{i}" for i in range(5)]
    answerable, no_relevant = pick_subsets(ids, n_answerable=2, n_no_relevant=0, seed=1)
    assert len(answerable) == 2
    assert no_relevant == []


def test_pick_subsets_too_many_answerable() -> None:
    with pytest.raises(ValueError, match="not enough ids for answerable"):
        pick_subsets(["q0", "q1"], n_answerable=3, n_no_relevant=0, seed=0)


def test_pick_subsets_too_many_no_relevant() -> None:
    with pytest.raises(ValueError, match="not enough ids for no-relevant"):
        pick_subsets(["q0", "q1", "q2"], n_answerable=2, n_no_relevant=2, seed=0)


# ---------------------------------------------------------------------------
# build_items
# ---------------------------------------------------------------------------


def test_build_items_answerable_and_no_relevant() -> None:
    d1_chunks = [make_chunk(f"c1{i}", "d1", i, f"text{i}") for i in range(6)]
    d2_chunks = [make_chunk(f"c2{i}", "d2", i, f"other{i}") for i in range(6)]
    retriever = FakeRetriever(
        {
            "claim A": [make_hit(c) for c in d1_chunks + d2_chunks],
            "claim B": [make_hit(c) for c in d1_chunks[:2] + d2_chunks],
        }
    )
    claims = {"qA": "claim A", "qB": "claim B"}
    gold = {"qA": {"d1"}, "qB": {"d1"}}

    items = build_items(
        retriever=retriever,
        claims=claims,
        gold=gold,
        answerable_ids=["qA"],
        no_relevant_ids=["qB"],
        k=3,
        depth=5,
    )

    assert ("claim A", 5) in retriever.calls
    assert ("claim B", 5) in retriever.calls

    i0 = items[0]
    assert i0.query_id == "qA"
    assert i0.kind == ItemKind.ANSWERABLE
    assert i0.hits == tuple(make_hit(c) for c in d1_chunks[:3])
    assert i0.question == format_question("claim A")
    assert i0.gold_doc_ids == ("d1",)

    i1 = items[1]
    assert i1.query_id == "qB"
    assert i1.kind == ItemKind.NO_RELEVANT
    assert len(i1.hits) == 3
    assert all(h.chunk.doc_id != "d1" for h in i1.hits)
    assert i1.gold_doc_ids == ("d1",)


def test_build_items_gold_sorted() -> None:
    chunk = make_chunk("c", "d1", 0, "hi")
    retriever = FakeRetriever({"claim": [make_hit(chunk)]})
    items = build_items(
        retriever=retriever,
        claims={"q": "claim"},
        gold={"q": {"d3", "d1", "d2"}},
        answerable_ids=["q"],
        no_relevant_ids=[],
        k=1,
        depth=1,
    )
    assert items[0].gold_doc_ids == ("d1", "d2", "d3")


def test_build_items_k_lt_1() -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        build_items(
            retriever=FakeRetriever(),
            claims={},
            gold={},
            answerable_ids=[],
            no_relevant_ids=[],
            k=0,
            depth=1,
        )


def test_build_items_depth_lt_k() -> None:
    with pytest.raises(ValueError, match="depth must be at least k"):
        build_items(
            retriever=FakeRetriever(),
            claims={},
            gold={},
            answerable_ids=[],
            no_relevant_ids=[],
            k=3,
            depth=2,
        )


def test_build_items_overlap() -> None:
    with pytest.raises(ValueError, match="ids appear in both subsets"):
        build_items(
            retriever=FakeRetriever(),
            claims={"q": "claim"},
            gold={"q": {"d1"}},
            answerable_ids=["q"],
            no_relevant_ids=["q"],
            k=1,
            depth=1,
        )


def test_build_items_all_hits_gold_for_no_relevant() -> None:
    chunks = [make_chunk(f"c{i}", "d1", i, f"t{i}") for i in range(3)]
    retriever = FakeRetriever({"claim B": [make_hit(c) for c in chunks]})
    with pytest.raises(ValueError, match="non-gold hits remain"):
        build_items(
            retriever=retriever,
            claims={"qB": "claim B"},
            gold={"qB": {"d1"}},
            answerable_ids=[],
            no_relevant_ids=["qB"],
            k=2,
            depth=3,
        )


# ---------------------------------------------------------------------------
# run_answer_eval
# ---------------------------------------------------------------------------


def test_run_answer_eval_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = [make_answerable_item(), make_no_relevant_item()]
    fake = FakeHTTP([envelope(answerable_content()), envelope(abstain_content())])
    monkeypatch.setattr("urllib.request.urlopen", fake)

    seen: list[tuple[int, str]] = []

    def on_item(index: int, record: AnswerEvalRecord) -> None:
        seen.append((index, record.query_id))

    records = run_answer_eval(items, make_generator(tmp_path), max_sentences=3, on_item=on_item)

    assert len(records) == 2
    assert seen == [(0, "q0"), (1, "q1")]

    r0, r1 = records

    assert r0.query_id == "q0"
    assert r0.kind == ItemKind.ANSWERABLE
    assert r0.error is None
    assert r0.response is not None
    assert isinstance(r0.response, dict)
    assert r0.response["abstained"] is False
    assert r0.grade.parse_ok is True
    assert r0.grade.abstained is False
    assert r0.grade.n_raw_citations == 1
    assert r0.grade.n_valid_citations == 1
    assert r0.grade.n_valid_gold_citations == 1

    assert r1.query_id == "q1"
    assert r1.kind == ItemKind.NO_RELEVANT
    assert r1.error is None
    assert r1.response is not None
    assert r1.grade.parse_ok is True
    assert r1.grade.abstained is True
    assert r1.grade.abstention_correct is True

    for record in records:
        for ctx in record.contexts:
            assert set(ctx.keys()) == {"index", "doc_id", "chunk_id", "text_sha256"}

    assert r0.contexts == (
        {
            "index": 0,
            "doc_id": "d1",
            "chunk_id": "c0",
            "text_sha256": hashlib.sha256(TEXT0.encode()).hexdigest(),
        },
    )

    assert answerable_content() in r0.raw_contents
    assert abstain_content() in r1.raw_contents


def test_run_answer_eval_generator_error_http_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    headers = email.message.Message()
    err = urllib.error.HTTPError(BASE_URL, 500, "boom", headers, None)
    fake = FakeHTTP([err])
    monkeypatch.setattr("urllib.request.urlopen", fake)

    records = run_answer_eval([make_answerable_item()], make_generator(tmp_path))

    assert len(records) == 1
    record = records[0]
    assert record.response is None
    assert record.error is not None
    assert record.grade.parse_ok is False


def test_run_answer_eval_cache_miss_replay_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeHTTP([])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    generator = make_generator(tmp_path, replay_only=True)

    with pytest.raises(CacheMissError):
        run_answer_eval([make_answerable_item()], generator)
    assert fake.calls == []


def test_record_replay_round_trip_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [make_answerable_item(), make_no_relevant_item()]

    fake1 = FakeHTTP([envelope(answerable_content()), envelope(abstain_content())])
    monkeypatch.setattr("urllib.request.urlopen", fake1)
    records1 = run_answer_eval(items, make_generator(tmp_path, replay_only=False))
    assert len(fake1.calls) == 2

    fake2 = FakeHTTP([])
    monkeypatch.setattr("urllib.request.urlopen", fake2)
    records2 = run_answer_eval(items, make_generator(tmp_path, replay_only=True))

    assert fake2.calls == []
    assert [r.model_dump(mode="json") for r in records2] == [
        r.model_dump(mode="json") for r in records1
    ]


# ---------------------------------------------------------------------------
# build_answer_report / write_answer_report
# ---------------------------------------------------------------------------


def test_build_answer_report_keys_metrics_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _run_basic(tmp_path, monkeypatch)

    report = build_answer_report(
        records,
        config={"k": 3, "depth": 5},
        created_at="2024-01-01T00:00:00Z",
        git_sha="abc123",
        git_dirty=False,
        dataset_name="scifact",
        dataset_checksum="deadbeef",
        n_cache_hits=0,
        n_api_calls=2,
    )

    assert report["schema_version"] == ANSWER_EVAL_SCHEMA_VERSION
    assert report["kind"] == "answer_eval"
    assert report["selection_method"] == SELECTION_METHOD
    assert report["question_template"] == QUESTION_TEMPLATE
    assert report["config"] == {"k": 3, "depth": 5}
    assert report["git_sha"] == "abc123"
    assert report["git_dirty"] is False
    assert report["dataset_name"] == "scifact"
    assert report["dataset_checksum"] == "deadbeef"
    assert report["n_cache_hits"] == 0
    assert report["n_api_calls"] == 2
    assert isinstance(report["metrics"], dict)
    assert len(report["records"]) == len(records)
    assert report["records"][0]["query_id"] == "q0"


def test_build_answer_report_empty_raises() -> None:
    with pytest.raises(ValueError):
        build_answer_report(
            [],
            config={},
            created_at="t",
            git_sha="s",
            git_dirty=None,
            dataset_name="d",
            dataset_checksum="c",
            n_cache_hits=0,
            n_api_calls=0,
        )


def test_write_answer_report_writes_json_and_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "sub" / "report.json"
    report = {"a": 1, "b": ["x", "y"], "c": None}

    result = write_answer_report(report, path)

    assert result == path
    assert path.parent.is_dir()
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == report


def test_build_items_requires_k_hits_for_answerable() -> None:
    chunk = make_chunk("c", "d1", 0, "hi")
    retriever = FakeRetriever({"claim": [make_hit(chunk)]})
    with pytest.raises(ValueError, match="only 1 hits"):
        build_items(
            retriever=retriever,
            claims={"q": "claim"},
            gold={"q": {"d1"}},
            answerable_ids=["q"],
            no_relevant_ids=[],
            k=2,
            depth=2,
        )
