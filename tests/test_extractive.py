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


# --- Markdown/HTML heading lines are never part of an extracted sentence ---

MARKDOWN_CHUNK = (
    "# Comparison Sorting Algorithms\n\n## Quicksort\n\n"
    "Quicksort partitions around a pivot and recurses on both sides."
)


def test_markdown_heading_lines_are_stripped_from_quotes_with_exact_offsets() -> None:
    # Regression for the Docker smoke test answer, which started with
    # "# Comparison Sorting Algorithms\n\n## Quicksort\n\nQuicksort partitions...".
    hit = make_hit(MARKDOWN_CHUNK, start_char=40, format="markdown")
    response = ExtractiveAnswerGenerator().generate(
        query="How does quicksort choose a pivot?", hits=[hit]
    )

    quote = "Quicksort partitions around a pivot and recurses on both sides."
    assert response.answer == quote
    (citation,) = response.citations
    assert citation.quote == quote
    local = MARKDOWN_CHUNK.index(quote)
    assert (citation.start_char, citation.end_char) == (40 + local, 40 + local + len(quote))


def test_html_chunks_strip_headings_too() -> None:
    # The HTML loader normalizes <h1>-<h6> to ATX heading lines.
    hit = make_hit("## Pivot Choice\nMedian of three picks the pivot.", format="html")
    response = ExtractiveAnswerGenerator().generate(query="pivot choice", hits=[hit])

    assert response.answer == "Median of three picks the pivot."


def test_heading_only_chunk_abstains() -> None:
    # Last line has no trailing newline: the heading runs to the end of the chunk.
    hit = make_hit("# Quicksort\n\n## Pivot", format="markdown")
    response = ExtractiveAnswerGenerator().generate(query="quicksort pivot", hits=[hit])

    assert response.abstained is True
    assert response.citations == ()


def test_heading_is_a_sentence_boundary_between_unpunctuated_lines() -> None:
    text = "Pivot selection matters\n### Lomuto scheme\nLomuto uses the last pivot element"
    hit = make_hit(text, start_char=7, format="markdown")
    response = ExtractiveAnswerGenerator().generate(query="pivot", hits=[hit])

    assert [c.quote for c in response.citations] == [
        "Pivot selection matters",
        "Lomuto uses the last pivot element",
    ]
    for citation in response.citations:
        local = citation.start_char - 7
        assert text[local : citation.end_char - 7] == citation.quote
    assert all("#" not in c.quote for c in response.citations)


def test_hash_lines_inside_fenced_code_are_not_headings() -> None:
    text = "```bash\n# choose pivot index\n```\n## Pivot\nThe pivot splits the array."
    hit = make_hit(text, format="markdown")
    response = ExtractiveAnswerGenerator().generate(query="pivot", hits=[hit], max_sentences=5)

    quotes = [c.quote for c in response.citations]
    assert quotes == ["```bash\n# choose pivot index\n```", "The pivot splits the array."]


def test_plain_text_and_unknown_formats_keep_hash_lines() -> None:
    text = "# pivot notes\nThe pivot splits the array."
    for meta in ({"format": "text"}, {}):
        hit = make_hit(text, **meta)
        response = ExtractiveAnswerGenerator().generate(query="pivot", hits=[hit])
        assert response.answer == text


def test_heading_stripping_preserves_repeated_sentence_offsets() -> None:
    text = "## A\nPivot first. Pivot first.\n## B\nPivot first."
    hit = make_hit(text, format="markdown")
    response = ExtractiveAnswerGenerator().generate(query="pivot", hits=[hit])

    # Deduped to one quote, at the first occurrence.
    (citation,) = response.citations
    assert citation.quote == "Pivot first."
    assert (citation.start_char, citation.end_char) == (5, 17)
