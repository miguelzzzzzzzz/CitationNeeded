from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_engine.eval.runner import EvalConfig, retrieve_documents, run_evaluation
from rag_engine.models import Chunk, Document
from rag_engine.retrieval.retriever import RetrievedChunk
from rag_engine.retrieval.vector_store import Filters

# A 12-token paragraph that is nothing but the query term.
HOG_PARAGRAPH: str = " ".join(["zebra"] * 12)
# A short document that mentions "zebra" exactly once among other words.
SHORT_TEXT: str = "the quick brown fox spots one zebra and then trots away home"

Corpus = tuple[list[Document], dict[str, str], dict[str, dict[str, int]]]


def _make_zebra_corpus() -> Corpus:
    """One hog document matching 'zebra' in every chunk plus five short uniques."""
    hog = Document.create(
        source="hog",
        title="Hog",
        format="txt",
        text="\n\n".join([HOG_PARAGRAPH] * 12),
        corpus_id="t",
    )
    shorts = [
        Document.create(
            source=f"short-{index}",
            title=f"Short {index}",
            format="txt",
            text=SHORT_TEXT,
            corpus_id="t",
        )
        for index in range(5)
    ]
    documents = [hog, *shorts]
    queries = {"zebra": "zebra"}
    qrels: dict[str, dict[str, int]] = {"zebra": {shorts[0].doc_id: 1}}
    return documents, queries, qrels


@pytest.fixture
def zebra_corpus() -> Corpus:
    return _make_zebra_corpus()


class _StubRetriever:
    """Returns the first ``k`` entries of a fixed ranked list and records every ``k``."""

    name: str = "stub"

    def __init__(self, doc_ids: Sequence[str]) -> None:
        self._doc_ids: tuple[str, ...] = tuple(doc_ids)
        self.calls: list[int] = []

    def retrieve(
        self,
        query: str,
        k: int = 10,
        filters: Filters | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(k)
        return [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id=f"{doc_id}-{rank}",
                    doc_id=doc_id,
                    index=rank - 1,
                    text="zebra",
                    start_char=0,
                    end_char=5,
                    token_count=1,
                ),
                score=1.0 / rank,
                rank=rank,
                retriever=self.name,
            )
            for rank, doc_id in enumerate(self._doc_ids[:k], start=1)
        ]


def test_dominant_document_does_not_starve_unique_docs(zebra_corpus: Corpus) -> None:
    documents, queries, qrels = zebra_corpus
    config = EvalConfig(
        dataset="scifact",
        modes=("dense", "lexical", "hybrid"),
        embedder="hashing",
        top_k=4,
        depth=2,
        chunk_strategy="fixed",
        chunk_size=16,
        chunk_overlap=0,
        corpus_id="t",
    )
    report = run_evaluation(
        documents=documents,
        queries=queries,
        qrels=qrels,
        config=config,
        dataset_checksum="zebra-checksum",
        dataset_name="zebra-fixture",
    )

    assert report.n_queries == 1
    assert set(report.metrics) == {"dense", "lexical", "hybrid"}
    for mode in ("dense", "lexical", "hybrid"):
        metrics = report.metrics[mode]
        assert isinstance(metrics, dict)
        assert "recall@10" in metrics
        assert report.retrieval[mode]["initial_depth"] == 4.0
        assert report.retrieval[mode]["queries_short_of_k"] == 0.0
        assert report.retrieval[mode]["max_depth_used"] > 2


def test_capped_max_depth_starves_lexical_retrieval(zebra_corpus: Corpus) -> None:
    documents, queries, qrels = zebra_corpus
    config = EvalConfig(
        dataset="scifact",
        modes=("lexical",),
        embedder="hashing",
        top_k=4,
        depth=2,
        max_depth=4,
        chunk_strategy="fixed",
        chunk_size=16,
        chunk_overlap=0,
        corpus_id="t",
    )
    report = run_evaluation(
        documents=documents,
        queries=queries,
        qrels=qrels,
        config=config,
        dataset_checksum="zebra-checksum",
        dataset_name="zebra-fixture",
    )

    retrieval = report.retrieval["lexical"]
    assert report.n_queries == 1
    assert retrieval["initial_depth"] == 4.0
    assert retrieval["max_depth_used"] == 4.0
    assert retrieval["queries_short_of_k"] >= 1.0


def test_retrieve_documents_doubles_depth_and_reports_exhaustion() -> None:
    retriever = _StubRetriever(["a", "a", "a", "a", "b", "c", "d"])
    doc_ids, final_depth, exhausted = retrieve_documents(
        retriever, "zebra", k=3, depth=2, max_depth=100, reranking=False
    )
    assert doc_ids == ["a", "b", "c"]
    assert final_depth == 8
    assert exhausted is False
    assert retriever.calls == [2, 4, 8]

    short = _StubRetriever(["a", "a", "a", "a", "b"])
    short_docs, _, short_exhausted = retrieve_documents(
        short, "zebra", k=3, depth=2, max_depth=100, reranking=False
    )
    assert short_docs == ["a", "b"]
    assert short_exhausted is True
    assert short.calls == [2, 4, 8]
