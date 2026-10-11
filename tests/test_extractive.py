"""Tests for :mod:`rag_engine.answering.extractive`."""

from __future__ import annotations

import pytest

from rag_engine.answering.extractive import ExtractiveAnswerGenerator
from rag_engine.models import Chunk
from rag_engine.retrieval.retriever import RetrievedChunk


def make_hit(
    text: str,
    *,
    rank: int = 1,
    start_char: int = 0,
    doc_id: str = "doc",
    score: float = 1.0,
    **meta: object,
) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{doc_id}-0",
        doc_id=doc_id,
        index=0,
        text=text,
        start_char=start_char,
        end_char=start_char + len(text),
        token_count=len(text.split()),
        heading_path=("Intro",),
        page=1,
        metadata={"source": f"{doc_id}.md", "content_hash": "abc", "page_end": 1, **meta},
    )
    return RetrievedChunk(chunk=chunk, score=score, rank=rank, retriever="dense")


def test_selects_highest_overlap_sentence_with_document_offsets() -> None:
    hit = make_hit("Cats are great pets. Dogs are loyal companions.", start_char=100)
    response = ExtractiveAnswerGenerator().generate(query="loyal dogs", hits=[hit])

    assert response.answer == "Dogs are loyal companions."
    assert response.abstained is False
    assert len(response.citations) == 1
    citation = response.citations[0]
    assert citation.quote == "Dogs are loyal companions."
    # local offset of second sentence inside chunk text is 21
    assert citation.start_char == 121
    assert citation.end_char == 147
    assert citation.source == "doc.md"


def test_answer_joins_multiple_selected_sentences() -> None:
    hit = make_hit(
        "Cats are great pets. Dogs are loyal companions. Fish swim in tanks.",
        start_char=10,
    )
    response = ExtractiveAnswerGenerator().generate(
        query="cats pets dogs loyal", hits=[hit], max_sentences=2
    )

    assert "Cats are great pets." in response.answer
    assert "Dogs are loyal companions." in response.answer
    assert len(response.citations) == 2


def test_max_sentences_caps_selection() -> None:
    hit = make_hit("Alpha one. Beta two. Gamma three. Delta four.")
    response = ExtractiveAnswerGenerator().generate(
        query="alpha beta gamma delta", hits=[hit], max_sentences=2
    )

    assert len(response.citations) == 2
    assert not response.abstained


def test_dedupes_near_identical_sentences() -> None:
    hit = make_hit("Cats are great.  CATS   are GREAT. Dogs bark loudly.")
    response = ExtractiveAnswerGenerator().generate(query="cats great", hits=[hit], max_sentences=3)

    assert len(response.citations) == 1
    assert " ".join(response.answer.split()).casefold() == "cats are great."


def test_abstains_on_empty_hits() -> None:
    response = ExtractiveAnswerGenerator().generate(query="cats", hits=[])

    assert response.abstained is True
    assert response.answer == ""
    assert response.citations == ()


def test_abstains_when_no_sentence_overlaps_query() -> None:
    hit = make_hit("Dogs bark loudly at night.")
    response = ExtractiveAnswerGenerator().generate(query="quantum physics", hits=[hit])

    assert response.abstained is True
    assert response.answer == ""


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_blank_query_raises_value_error(query: str) -> None:
    with pytest.raises(ValueError, match="query"):
        ExtractiveAnswerGenerator().generate(query=query, hits=[make_hit("Cats are great pets.")])


@pytest.mark.parametrize("max_sentences", [0, -1])
def test_invalid_max_sentences_raises_value_error(max_sentences: int) -> None:
    with pytest.raises(ValueError, match="max_sentences"):
        ExtractiveAnswerGenerator().generate(
            query="cats", hits=[make_hit("Cats are great pets.")], max_sentences=max_sentences
        )


def test_stopword_only_query_abstains_without_error() -> None:
    hit = make_hit("Cats are great pets.")
    response = ExtractiveAnswerGenerator().generate(query="the and of to a", hits=[hit])

    assert response.abstained is True
    assert response.answer == ""


def test_prefers_higher_term_overlap() -> None:
    weak = make_hit("Alpha appears once here.", rank=1, doc_id="weak")
    strong = make_hit("Alpha beta omega together.", rank=2, doc_id="strong")

    response = ExtractiveAnswerGenerator().generate(query="alpha beta omega", hits=[weak, strong])

    assert response.citations[0].quote == "Alpha beta omega together."


def test_prefers_lower_rank_when_overlap_ties() -> None:
    later = make_hit("Alpha beta zeta.", rank=2, doc_id="later")
    earlier = make_hit("Alpha beta omega.", rank=1, doc_id="earlier")

    response = ExtractiveAnswerGenerator().generate(query="alpha beta", hits=[later, earlier])

    assert response.citations[0].chunk_id == "earlier-0"


def test_generator_name_is_extractive() -> None:
    assert ExtractiveAnswerGenerator().name == "extractive"
