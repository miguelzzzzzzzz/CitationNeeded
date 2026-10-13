"""Tests for the committed, replayable answer-generation cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.eval.answer_cache import (
    CACHE_SCHEMA_VERSION,
    CachedAnswerGenerator,
    CacheMissError,
    ResponseCache,
    cache_key,
)
from rag_engine.models import Chunk
from rag_engine.retrieval.retriever import RetrievedChunk


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeHTTP:
    def __init__(self, responses: list[bytes | Exception]) -> None:
        self.responses = responses
        self.calls: list[object] = []

    def __call__(self, req: object, timeout: float | None = None) -> FakeResponse:
        self.calls.append(req)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)


def envelope(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


def make_hit(text: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
        metadata={"source": "test"},
    )
    return RetrievedChunk(chunk=chunk, score=1.0, rank=1, retriever="fake")


def valid_content(quote: str) -> str:
    return json.dumps(
        {
            "answer": f"Answer for {quote}",
            "citations": [{"chunk_index": 0, "quote": quote}],
            "abstained": False,
        }
    )


def test_cache_key_deterministic_and_sensitive() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    base = cache_key(
        base_url="https://a.test/v1", model="m", messages=msgs, temperature=0.0, max_tokens=10
    )
    assert base == cache_key(
        base_url="https://a.test/v1", model="m", messages=msgs, temperature=0.0, max_tokens=10
    )
    assert base != cache_key(
        base_url="https://a.test/v1", model="m2", messages=msgs, temperature=0.0, max_tokens=10
    )
    changed = [{"role": "user", "content": "bye"}]
    assert base != cache_key(
        base_url="https://a.test/v1", model="m", messages=changed, temperature=0.0, max_tokens=10
    )
    assert base != cache_key(
        base_url="https://a.test/v1", model="m", messages=msgs, temperature=0.1, max_tokens=10
    )
    assert base != cache_key(
        base_url="https://a.test/v1", model="m", messages=msgs, temperature=0.0, max_tokens=11
    )
    assert base != cache_key(
        base_url="https://b.test/v1", model="m", messages=msgs, temperature=0.0, max_tokens=10
    )
    # Trailing slash and int-vs-float temperature do not change the request identity.
    assert base == cache_key(
        base_url="https://a.test/v1/", model="m", messages=msgs, temperature=0, max_tokens=10
    )


def test_cache_loads_valid_jsonl_with_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    record = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "key": "k",
        "model": "m",
        "content": "c",
    }
    path.write_text(f"\n{json.dumps(record)}\n\n", encoding="utf-8")
    cache = ResponseCache(path)
    assert len(cache) == 1
    assert "k" in cache
    assert cache.get("k") == "c"
    assert cache.get("missing") is None


def test_cache_load_invalid_json_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"invalid JSON on line 1"):
        ResponseCache(path)


def test_cache_load_non_object_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 1 is not a JSON object"):
        ResponseCache(path)


def test_cache_load_wrong_schema_version_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    record = {"schema_version": 999, "key": "k", "model": "m", "content": "c"}
    path.write_text(f"{json.dumps(record)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 1 has schema_version 999"):
        ResponseCache(path)


def test_cache_load_non_str_key_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    record = {"schema_version": CACHE_SCHEMA_VERSION, "key": 1, "model": "m", "content": "c"}
    path.write_text(f"{json.dumps(record)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 1 has missing/non-str key"):
        ResponseCache(path)


def test_cache_load_non_str_model_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    record = {"schema_version": CACHE_SCHEMA_VERSION, "key": "k", "model": 1, "content": "c"}
    path.write_text(f"{json.dumps(record)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 1 has missing/non-str model"):
        ResponseCache(path)


def test_cache_load_non_str_content_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    record = {"schema_version": CACHE_SCHEMA_VERSION, "key": "k", "model": "m", "content": 1}
    path.write_text(f"{json.dumps(record)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 1 has missing/non-str content"):
        ResponseCache(path)


def test_cache_load_conflicting_duplicate_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    first = {"schema_version": CACHE_SCHEMA_VERSION, "key": "k", "model": "m", "content": "a"}
    second = {"schema_version": CACHE_SCHEMA_VERSION, "key": "k", "model": "m", "content": "b"}
    path.write_text(f"{json.dumps(first)}\n{json.dumps(second)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2 duplicates key 'k' with different content"):
        ResponseCache(path)


def test_cache_load_tolerates_identical_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    record = {"schema_version": CACHE_SCHEMA_VERSION, "key": "k", "model": "m", "content": "a"}
    path.write_text(f"{json.dumps(record)}\n{json.dumps(record)}\n", encoding="utf-8")
    cache = ResponseCache(path)
    assert len(cache) == 1
    assert cache.get("k") == "a"


def test_cache_put_appends_creates_parents_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "cache.jsonl"
    cache = ResponseCache(path)
    cache.put("k", model="m", content="c")
    assert path.exists()
    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert len(cache) == 1
    assert "k" in cache
    assert cache.get("k") == "c"
    cache.put("k", model="m", content="c")
    assert path.read_text(encoding="utf-8").count("\n") == 1
    with pytest.raises(ValueError, match=r"already cached with different content"):
        cache.put("k", model="m", content="other")


def test_reloaded_cache_sees_put_entries(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = ResponseCache(path)
    cache.put("k", model="m", content="c")
    reloaded = ResponseCache(path)
    assert len(reloaded) == 1
    assert "k" in reloaded
    assert reloaded.get("k") == "c"


def test_cached_generator_record_mode_calls_http_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "Alpha beta."
    hit = make_hit(text)
    content = valid_content(text)
    fake = FakeHTTP([envelope(content)])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    cache = ResponseCache(tmp_path / "cache.jsonl")
    gen = CachedAnswerGenerator(
        cache=cache,
        replay_only=False,
        base_url="http://example.invalid",
        model="m",
    )
    response = gen.generate("q", [hit])
    assert gen.n_api_calls == 1
    assert gen.n_cache_hits == 0
    assert gen.raw_contents == [content]
    assert gen.last_raw_content == content
    assert response.abstained is False
    assert len(cache) == 1
    assert len(fake.calls) == 1


def test_cached_generator_replay_only_returns_cached_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "Alpha beta."
    hit = make_hit(text)
    content = valid_content(text)
    path = tmp_path / "cache.jsonl"
    cache = ResponseCache(path)
    monkeypatch.setattr("urllib.request.urlopen", FakeHTTP([envelope(content)]))
    recorder = CachedAnswerGenerator(
        cache=cache,
        replay_only=False,
        base_url="http://example.invalid",
        model="m",
    )
    first = recorder.generate("q", [hit])
    assert recorder.n_api_calls == 1

    reloaded = ResponseCache(path)
    fake_replay = FakeHTTP([])
    monkeypatch.setattr("urllib.request.urlopen", fake_replay)
    replayer = CachedAnswerGenerator(
        cache=reloaded,
        replay_only=True,
        base_url="http://example.invalid",
        model="m",
    )
    second = replayer.generate("q", [hit])
    assert second == first
    assert replayer.n_api_calls == 0
    assert replayer.n_cache_hits == 1
    assert replayer.raw_contents == [content]
    assert fake_replay.calls == []


def test_replay_only_miss_raises_cache_miss_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hit = make_hit("Alpha.")
    cache = ResponseCache(tmp_path / "cache.jsonl")
    monkeypatch.setattr("urllib.request.urlopen", FakeHTTP([]))
    gen = CachedAnswerGenerator(
        cache=cache,
        replay_only=True,
        base_url="http://example.invalid",
        model="m",
    )
    with pytest.raises(CacheMissError) as excinfo:
        gen.generate("q", [hit])
    assert isinstance(excinfo.value, LookupError)


def test_repair_flow_records_and_caches_both_raw_replies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "Alpha beta."
    hit = make_hit(text)
    invalid = "not json"
    valid = valid_content(text)
    fake = FakeHTTP([envelope(invalid), envelope(valid)])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    cache = ResponseCache(tmp_path / "cache.jsonl")
    gen = CachedAnswerGenerator(
        cache=cache,
        replay_only=False,
        base_url="http://example.invalid",
        model="m",
        max_repair_attempts=1,
    )
    response = gen.generate("q", [hit])
    assert response.abstained is False
    assert gen.n_api_calls == 2
    assert gen.raw_contents == [invalid, valid]
    assert len(cache) == 2
    assert len(fake.calls) == 2


def test_raw_contents_reset_between_generate_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hit1 = make_hit("Alpha.")
    hit2 = make_hit("Beta.")
    content1 = valid_content("Alpha.")
    content2 = valid_content("Beta.")
    fake = FakeHTTP([envelope(content1), envelope(content2)])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    cache = ResponseCache(tmp_path / "cache.jsonl")
    gen = CachedAnswerGenerator(
        cache=cache,
        replay_only=False,
        base_url="http://example.invalid",
        model="m",
    )
    gen.generate("q1", [hit1])
    assert gen.raw_contents == [content1]
    gen.generate("q2", [hit2])
    assert gen.raw_contents == [content2]


def test_last_raw_content_none_before_and_after_empty_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = ResponseCache(tmp_path / "cache.jsonl")
    monkeypatch.setattr("urllib.request.urlopen", FakeHTTP([]))
    gen = CachedAnswerGenerator(
        cache=cache,
        replay_only=False,
        base_url="http://example.invalid",
        model="m",
    )
    assert gen.last_raw_content is None
    response = gen.generate("q", [])
    assert response.abstained is True
    assert gen.raw_contents == []
    assert gen.last_raw_content is None
    assert gen.n_api_calls == 0
