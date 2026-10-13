"""Tests for :mod:`rag_engine.answering.openai_client` (stdlib urllib; no network)."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from rag_engine.answering.openai_client import OpenAICompatibleAnswerGenerator
from rag_engine.models import Chunk
from rag_engine.retrieval.retriever import RetrievedChunk

BASE_URL = "https://api.example.test/v1"
MODEL = "test-model"


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
        self._responses = list(responses)
        self.calls: list[Request] = []

    def __call__(self, req: Request, timeout: float | None = None) -> FakeResponse:
        self.calls.append(req)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


def make_hit(text: str, *, start_char: int = 0, rank: int = 1) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="doc-0",
        doc_id="doc",
        index=0,
        text=text,
        start_char=start_char,
        end_char=start_char + len(text),
        token_count=len(text.split()),
        heading_path=("Intro",),
        page=1,
        metadata={"source": "doc.md", "content_hash": "abc", "page_end": 1},
    )
    return RetrievedChunk(chunk=chunk, score=1.0, rank=rank, retriever="dense")


def chat_envelope(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


def make_generator(**overrides: Any) -> OpenAICompatibleAnswerGenerator:
    kwargs: dict[str, Any] = {
        "base_url": BASE_URL,
        "model": MODEL,
        "api_key": None,
        "timeout_s": 5.0,
        "max_tokens": 128,
        "temperature": 0.0,
        "max_repair_attempts": 1,
    }
    kwargs.update(overrides)
    return OpenAICompatibleAnswerGenerator(**kwargs)


def test_constructor_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        make_generator(base_url="  ")
    with pytest.raises(ValueError):
        make_generator(model="")
    with pytest.raises(ValueError):
        make_generator(timeout_s=0)
    with pytest.raises(ValueError):
        make_generator(max_tokens=0)
    with pytest.raises(ValueError):
        make_generator(temperature=-0.1)
    with pytest.raises(ValueError):
        make_generator(max_repair_attempts=-1)


def test_generate_rejects_blank_query_and_bad_max_sentences() -> None:
    gen = make_generator()
    with pytest.raises(ValueError, match="query"):
        gen.generate("  ", [make_hit("text")])
    with pytest.raises(ValueError, match="max_sentences"):
        gen.generate("cats", [make_hit("text")], max_sentences=0)


def test_empty_hits_abstain() -> None:
    response = make_generator().generate("cats", [])
    assert response.abstained is True
    assert response.model_name == MODEL
    assert response.generator == "openai"


def test_happy_path_grounds_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Cats sleep a lot. Dogs chase balls."
    hit = make_hit(text, start_char=50)
    payload = {
        "answer": "Cats sleep a lot.",
        "citations": [{"chunk_index": 0, "quote": "Cats sleep a lot."}],
        "abstained": False,
    }
    fake = FakeHTTP([chat_envelope(json.dumps(payload))])
    monkeypatch.setattr("urllib.request.urlopen", fake)

    response = make_generator().generate("Do cats sleep?", [hit])
    assert response.answer == "Cats sleep a lot."
    assert response.generator == "openai"
    assert response.model_name == MODEL
    assert len(response.citations) == 1
    citation = response.citations[0]
    assert citation.quote == "Cats sleep a lot."
    assert citation.start_char == 50
    assert citation.end_char == 50 + len("Cats sleep a lot.")
    assert citation.source == "doc.md"


def test_accepts_markdown_fenced_json(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = make_hit("Cats sleep a lot.")
    inner = json.dumps(
        {
            "answer": "Cats sleep a lot.",
            "citations": [{"chunk_index": 0, "quote": "Cats sleep a lot."}],
            "abstained": False,
        }
    )
    fenced = f"```json\n{inner}\n```"
    fake = FakeHTTP([chat_envelope(fenced)])
    monkeypatch.setattr("urllib.request.urlopen", fake)

    response = make_generator().generate("cats", [hit])
    assert response.answer == "Cats sleep a lot."
    assert not response.abstained


def test_repairs_malformed_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = make_hit("Cats sleep a lot.")
    good = json.dumps(
        {
            "answer": "Cats sleep a lot.",
            "citations": [{"chunk_index": 0, "quote": "Cats sleep a lot."}],
            "abstained": False,
        }
    )
    fake = FakeHTTP([chat_envelope("NOT JSON"), chat_envelope(good)])
    monkeypatch.setattr("urllib.request.urlopen", fake)

    response = make_generator(max_repair_attempts=1).generate("cats", [hit])
    assert response.answer == "Cats sleep a lot."
    assert len(fake.calls) == 2


def test_exhausted_repairs_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = make_hit("Cats sleep a lot.")
    fake = FakeHTTP([chat_envelope("nope"), chat_envelope("still nope")])
    monkeypatch.setattr("urllib.request.urlopen", fake)

    with pytest.raises(ValueError, match="unusable"):
        make_generator(max_repair_attempts=1).generate("cats", [hit])


def test_http_and_url_errors_become_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = make_hit("Cats sleep a lot.")
    http_err = HTTPError(BASE_URL, 500, "boom", None, None)  # type: ignore[arg-type]
    fake = FakeHTTP([http_err])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    with pytest.raises(ValueError, match="HTTP 500"):
        make_generator().generate("cats", [hit])

    fake2 = FakeHTTP([URLError("down")])
    monkeypatch.setattr("urllib.request.urlopen", fake2)
    with pytest.raises(ValueError, match="unreachable"):
        make_generator().generate("cats", [hit])


def test_unknown_quote_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = make_hit("Cats sleep a lot.")
    payload = {
        "answer": "Cats sleep a lot.",
        "citations": [{"chunk_index": 0, "quote": "not in the chunk"}],
        "abstained": False,
    }
    fake = FakeHTTP([chat_envelope(json.dumps(payload))])
    monkeypatch.setattr("urllib.request.urlopen", fake)

    response = make_generator().generate("cats", [hit])
    assert response.answer == "Cats sleep a lot."
    assert response.citations == ()


def test_authorization_header_only_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = make_hit("Cats sleep a lot.")
    payload = {"answer": "", "citations": [], "abstained": True}

    fake = FakeHTTP([chat_envelope(json.dumps(payload))])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    make_generator(api_key=None).generate("cats", [hit])
    assert "Authorization" not in fake.calls[0].headers

    fake2 = FakeHTTP([chat_envelope(json.dumps(payload))])
    monkeypatch.setattr("urllib.request.urlopen", fake2)
    make_generator(api_key="sk-test").generate("cats", [hit])
    assert fake2.calls[0].headers.get("Authorization") == "Bearer sk-test"


def test_name_property() -> None:
    assert make_generator().name == "openai"


def test_timeout_error_becomes_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP([TimeoutError()])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    with pytest.raises(ValueError, match="timed out"):
        make_generator().generate("cats", [make_hit("Cats sleep a lot.")])


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json",
        json.dumps(["list"]).encode(),
        json.dumps({"choices": []}).encode(),
        json.dumps({"choices": [{"message": {"content": 123}}]}).encode(),
    ],
)
def test_extract_content_rejects_bad_envelopes(monkeypatch: pytest.MonkeyPatch, raw: bytes) -> None:
    fake = FakeHTTP([raw])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    with pytest.raises(ValueError):
        make_generator().generate("cats", [make_hit("Cats sleep a lot.")])


@pytest.mark.parametrize(
    "content",
    [
        "```json\n\n```",
        "[]",
        json.dumps({"answer": "x", "citations": [], "abstained": "nope"}),
        json.dumps({"answer": 1, "citations": [], "abstained": False}),
        json.dumps({"answer": "ok", "citations": "nope", "abstained": False}),
    ],
)
def test_parse_and_to_answer_validation_errors(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    fake = FakeHTTP([chat_envelope(content)])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    with pytest.raises(ValueError):
        make_generator(max_repair_attempts=0).generate("cats", [make_hit("Cats sleep a lot.")])


def test_build_citations_skips_invalid_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = make_hit("Cats sleep a lot.")
    payload = {
        "answer": "Cats sleep a lot.",
        "citations": [
            "bad",
            {"chunk_index": True, "quote": "Cats sleep a lot."},
            {"chunk_index": -1, "quote": "Cats sleep a lot."},
            {"chunk_index": 0, "quote": ""},
            {"chunk_index": 0, "quote": "Cats sleep a lot."},
        ],
        "abstained": False,
    }
    fake = FakeHTTP([chat_envelope(json.dumps(payload))])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    response = make_generator().generate("cats", [hit])
    assert len(response.citations) == 1


def test_case_insensitive_quote_locate(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = make_hit("Cats sleep a lot.")
    payload = {
        "answer": "cats sleep a lot.",
        "citations": [{"chunk_index": 0, "quote": "cats sleep a lot."}],
        "abstained": False,
    }
    fake = FakeHTTP([chat_envelope(json.dumps(payload))])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    response = make_generator().generate("cats", [hit])
    assert response.citations[0].quote == "Cats sleep a lot."


def test_data_wrapped_envelope_is_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Gateways such as Cline return {"data": <OpenAI payload>}.
    payload = {
        "answer": "Cats sleep a lot.",
        "citations": [{"chunk_index": 0, "quote": "Cats sleep a lot."}],
        "abstained": False,
    }
    inner = json.loads(chat_envelope(json.dumps(payload)))
    fake = FakeHTTP([json.dumps({"data": inner, "success": True}).encode()])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    response = make_generator().generate("cats", [make_hit("Cats sleep a lot.")])
    assert response.answer == "Cats sleep a lot."
    assert [c.quote for c in response.citations] == ["Cats sleep a lot."]


def test_data_key_without_object_is_not_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP([json.dumps({"data": "nope"}).encode()])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    with pytest.raises(ValueError, match="no choices"):
        make_generator().generate("cats", [make_hit("Cats sleep a lot.")])
