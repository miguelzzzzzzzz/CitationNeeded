"""BM25 tests with scores worked out by hand.

Corpus (k1 = 1.2, b = 0.75), one chunk per document:

    d1 "apple banana apple"           |d| = 3
    d2 "banana cherry"                |d| = 2
    d3 "cherry date elderberry fig"   |d| = 4      N = 3, avgdl = 3

    idf(df=1) = ln(1 + 2.5/1.5) = ln(8/3) = 0.980829
    idf(df=2) = ln(1 + 1.5/2.5) = ln(1.6) = 0.470004

    length norm K(|d|) = 1 - b + b*|d|/avgdl:  K(3) = 1, K(2) = 0.75, K(4) = 1.25
    tf part = tf * 2.2 / (tf + 1.2 * K)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_engine.models import Chunk
from rag_engine.retrieval import BM25Index, tokenize

D1, D2, D3 = "apple banana apple", "banana cherry", "cherry date elderberry fig"


def chunk(doc: str, text: str, index: int = 0, **metadata: object) -> Chunk:
    return Chunk(
        chunk_id=f"{doc}#{index}",
        doc_id=doc,
        index=index,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
        metadata=dict(metadata),
    )


@pytest.fixture
def index() -> BM25Index:
    bm25 = BM25Index()
    bm25.upsert([chunk("d1", D1, lang="en"), chunk("d2", D2, lang="de"), chunk("d3", D3)])
    return bm25


def scores(bm25: BM25Index, query: str) -> dict[str, float]:
    return {hit.chunk.doc_id: hit.score for hit in bm25.search(query)}


def test_corpus_statistics(index: BM25Index) -> None:
    assert len(index) == 3
    assert index.average_length == pytest.approx(3.0)
    assert index.document_frequency("banana") == 2
    assert index.idf("apple") == pytest.approx(0.980829, abs=1e-6)
    assert index.idf("banana") == pytest.approx(0.470004, abs=1e-6)


def test_term_frequency_saturates(index: BM25Index) -> None:
    # d1: tf = 2, K = 1 -> 2 * 2.2 / 3.2 = 1.375; 0.980829 * 1.375 = 1.348640
    assert scores(index, "apple") == pytest.approx({"d1": 1.348640}, abs=1e-6)


def test_shorter_document_wins_on_equal_tf(index: BM25Index) -> None:
    # d2: 2.2 / (1 + 0.9) = 1.157895 -> 0.544215;  d1: 2.2 / 2.2 = 1 -> 0.470004
    hits = index.search("banana")
    assert [h.chunk.doc_id for h in hits] == ["d2", "d1"]
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].score == pytest.approx(0.544215, abs=1e-6)
    assert hits[1].score == pytest.approx(0.470004, abs=1e-6)


def test_multi_term_scores_add_up(index: BM25Index) -> None:
    # d3: cherry 0.470004 * 2.2/2.5 = 0.413603, date 0.980829 * 0.88 = 0.863130
    expected = {"d3": 1.276733, "d2": 0.544215}
    assert scores(index, "Cherry, DATE!") == pytest.approx(expected, abs=1e-6)


def test_repeated_query_terms_count_once(index: BM25Index) -> None:
    assert scores(index, "apple apple apple") == scores(index, "apple")


def test_b_zero_disables_length_normalisation_and_ties_keep_insertion_order() -> None:
    bm25 = BM25Index(b=0.0)
    bm25.upsert([chunk("d1", D1), chunk("d2", D2), chunk("d3", D3)])
    hits = bm25.search("banana")
    assert [h.chunk.doc_id for h in hits] == ["d1", "d2"]
    assert hits[0].score == pytest.approx(hits[1].score)
    assert hits[0].score == pytest.approx(0.470004, abs=1e-6)


def test_k1_zero_makes_scores_binary_in_tf() -> None:
    bm25 = BM25Index(k1=0.0)
    bm25.upsert([chunk("d1", D1), chunk("d2", D2), chunk("d3", D3)])
    assert scores(bm25, "apple") == pytest.approx({"d1": 0.980829}, abs=1e-6)


def test_no_match_and_empty_inputs() -> None:
    assert BM25Index().search("anything") == []
    bm25 = BM25Index()
    bm25.upsert([chunk("d1", D1)])
    assert bm25.search("zebra") == []
    assert bm25.search("  ... ") == []


def test_k_truncates_and_must_be_positive(index: BM25Index) -> None:
    assert [h.chunk.doc_id for h in index.search("banana cherry", k=1)] == ["d2"]
    with pytest.raises(ValueError, match="k must be positive"):
        index.search("apple", k=0)


def test_filters_apply_before_truncation(index: BM25Index) -> None:
    hits = index.search("banana", k=1, filters={"lang": "en"})
    assert [h.chunk.doc_id for h in hits] == ["d1"]
    assert index.search("banana", filters={"lang": ["fr"]}) == []


def test_upsert_replaces_whole_document_and_updates_statistics(index: BM25Index) -> None:
    index.upsert([chunk("d1", "kiwi", 0), chunk("d1", "kiwi mango", 1)])
    assert len(index) == 4
    assert index.search("apple") == []
    assert index.document_frequency("banana") == 1
    assert index.average_length == pytest.approx((1 + 2 + 2 + 4) / 4)
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        index.upsert([chunk("d9", "x"), chunk("d9", "y")])


def test_delete_document_removes_postings(index: BM25Index) -> None:
    assert index.delete_document("d2") == 1
    assert index.delete_document("missing") == 0
    assert index.document_frequency("banana") == 1
    assert index.document_frequency("cherry") == 1
    assert index.average_length == pytest.approx(3.5)
    # N = 2, df = 1 -> idf = ln(1 + 1.5/1.5) = ln 2
    assert index.idf("cherry") == pytest.approx(0.693147, abs=1e-6)


def test_heading_context_is_indexed() -> None:
    bm25 = BM25Index()
    with_header = chunk("d1", "body text").model_copy(update={"context_header": "Installation"})
    bm25.upsert([with_header])
    assert [h.chunk.doc_id for h in bm25.search("installation")] == ["d1"]


def test_save_and_load_round_trip(index: BM25Index, tmp_path: Path) -> None:
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert (loaded.k1, loaded.b, len(loaded)) == (1.2, 0.75, 3)
    assert scores(loaded, "cherry date") == scores(index, "cherry date")


def test_load_rejects_bad_files(index: BM25Index, tmp_path: Path) -> None:
    index.save(tmp_path)
    with pytest.raises(ValueError, match="inconsistent"):
        BM25Index.load(tmp_path, tokenizer=lambda text: text.split()[:1])
    (tmp_path / "bm25.json").write_text('{"format_version": 99}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported BM25 format"):
        BM25Index.load(tmp_path)


def test_parameter_validation() -> None:
    with pytest.raises(ValueError, match="k1"):
        BM25Index(k1=-1)
    with pytest.raises(ValueError, match="b must"):
        BM25Index(b=1.5)


def test_tokenize_normalises_and_filters_stopwords() -> None:
    fullwidth = "\uff26\uff55\uff4c\uff4c"  # NFKC maps this to "Full"
    assert tokenize(fullwidth + "-width CAFÉ, naïve_test 42") == [
        "full",
        "width",
        "café",
        "naïve_test",
        "42",
    ]
    assert tokenize("the cat and the hat", stopwords={"the", "and"}) == ["cat", "hat"]
