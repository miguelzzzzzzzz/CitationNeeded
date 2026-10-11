"""Tests for :mod:`rag_engine.answering.models`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_engine.answering.models import (
    ANSWER_SCHEMA_VERSION,
    AnswerResponse,
    Citation,
    citation_from_provenance,
)


def _citation(**overrides: object) -> Citation:
    data: dict[str, object] = {
        "chunk_id": "chunk-1",
        "doc_id": "doc-1",
        "quote": "hello world",
        "start_char": 0,
        "end_char": 11,
        "heading_path": ("Intro",),
        "generator": "extractive",  # ignored by Citation
    }
    data.pop("generator", None)
    data.update(overrides)
    return Citation.model_validate(data)


def _response(**overrides: object) -> AnswerResponse:
    data: dict[str, object] = {
        "query": "what colour is the sky?",
        "answer": "blue",
        "citations": (_citation(),),
        "generator": "extractive",
        "abstained": False,
    }
    data.update(overrides)
    return AnswerResponse.model_validate(data)


def test_citation_happy_path() -> None:
    citation = _citation(page=2, page_end=3)
    assert citation.chunk_id == "chunk-1"
    assert citation.quote == "hello world"
    assert citation.heading_path == ("Intro",)
    assert citation.page == 2
    assert citation.page_end == 3


def test_answer_response_happy_path() -> None:
    response = _response()
    assert response.schema_version == ANSWER_SCHEMA_VERSION
    assert response.query == "what colour is the sky?"
    assert response.answer == "blue"
    assert response.generator == "extractive"
    assert len(response.citations) == 1


def test_citation_from_provenance_converts_heading_path_list_to_tuple() -> None:
    citation = citation_from_provenance(
        doc_id="doc-9",
        chunk_id="chunk-9",
        quote="some source text",
        start_char=4,
        end_char=20,
        heading_path=["Chapter 1", "Section 2"],
    )
    assert citation.heading_path == ("Chapter 1", "Section 2")
    assert citation.quote == "some source text"


@pytest.mark.parametrize("quote", ["", "   ", "\n\t"])
def test_citation_rejects_blank_quote(quote: str) -> None:
    with pytest.raises(ValidationError):
        _citation(quote=quote)


@pytest.mark.parametrize(("start", "end"), [(5, 5), (9, 3), (0, 0)])
def test_citation_rejects_end_char_not_after_start_char(start: int, end: int) -> None:
    with pytest.raises(ValidationError):
        _citation(start_char=start, end_char=end)


@pytest.mark.parametrize("field", ["page", "page_end"])
@pytest.mark.parametrize("value", [0, -1])
def test_citation_rejects_page_below_one(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        _citation(**{field: value})


@pytest.mark.parametrize("query", ["", "  ", "\t\n"])
def test_answer_response_rejects_blank_query(query: str) -> None:
    with pytest.raises(ValidationError):
        _response(query=query)


@pytest.mark.parametrize("answer", ["", "   "])
def test_answer_response_rejects_empty_answer_when_not_abstained(answer: str) -> None:
    with pytest.raises(ValidationError):
        _response(answer=answer, abstained=False)


def test_answer_response_rejects_abstained_with_citations() -> None:
    with pytest.raises(ValidationError):
        _response(answer="", citations=(_citation(),), abstained=True)


def test_answer_response_rejects_wrong_schema_version() -> None:
    with pytest.raises(ValidationError):
        _response(schema_version=ANSWER_SCHEMA_VERSION + 1)


def test_abstained_response_allows_empty_answer() -> None:
    response = _response(answer="", citations=(), abstained=True)
    assert response.abstained is True
    assert response.answer == ""
