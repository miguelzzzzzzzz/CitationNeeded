"""Tests for cross-encoder reranking (``rag_engine.retrieval.rerank``)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pytest

from rag_engine.models import Chunk
from rag_engine.retrieval.embedders import HashingEmbedder
from rag_engine.retrieval.rerank import (
    CrossEncoderModel,
    CrossEncoderReranker,
    RerankingRetriever,
)
from rag_engine.retrieval.retriever import DenseRetriever, LexicalRetriever, RetrievedChunk
from rag_engine.retrieval.vector_store import Filters


class _StandInCrossEncoder:
    """Offline stand-in for ``fastembed.TextCrossEncoder`` (records every call).

    Only ONNX inference is replaced: with no fixed ``scores`` it counts how many
    distinct query words appear in each document.
    """

    def __init__(self, scores: Sequence[float] | None = None) -> None:
        self.calls: list[tuple[str, list[str], int]] = []
        self._scores: list[float] | None = None if scores is None else list(scores)

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 32) -> list[float]:
        docs = list(documents)
        self.calls.append((query, docs, batch_size))
        if self._scores is not None:
            return list(self._scores)
        terms = set(query.lower().split())
        return [float(sum(1 for term in terms if term in doc.lower())) for doc in docs]


class _FixedReranker:
    """``Reranker`` returning a fixed score per text (unknown texts score 0)."""

    def __init__(self, scores: Mapping[str, float] | None = None) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self._scores: dict[str, float] = dict(scores or {})

    @property
    def name(self) -> str:
        return "fixed"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        return [self._scores.get(text, 0.0) for text in texts]


class _ShortReranker:
    """Returns one score too few, to exercise the mismatch guard."""

    @property
    def name(self) -> str:
        return "short"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [1.0] * (len(texts) - 1)


class _RecordingBase:
    """First-stage retriever that records calls and replays fixed hits."""

    def __init__(self, results: Sequence[RetrievedChunk], name: str = "base") -> None:
        self._results = list(results)
        self._name = name
        self.calls: list[tuple[str, int, Filters | None]] = []

    @property
    def name(self) -> str:
        return self._name

    def retrieve(
        self, query: str, k: int = 10, filters: Filters | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((query, k, filters))
        return self._results[:k]


_CHUNKS: tuple[tuple[str, str], ...] = (
    ("c1", "bm25 ranks documents using term frequency and inverse document frequency"),
    ("c2", "bm25 applies document length normalization to damp long documents"),
    ("c3", "the cat sat on the mat"),
    ("c4", "bm25 is a lexical ranking function used for keyword search"),
    ("c5", "dense vectors encode meaning in a fixed number of dimensions"),
)


def _chunk(chunk_id: str, text: str, doc_id: str = "doc-1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
        metadata={"source": f"{doc_id}.md"},
    )


def _lexical_retriever() -> LexicalRetriever:
    retriever = LexicalRetriever()
    retriever.index(
        [_chunk(chunk_id, text, doc_id=f"doc-{chunk_id}") for chunk_id, text in _CHUNKS]
    )
    return retriever


def _hit(chunk_id: str, text: str, score: float, rank: int) -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(chunk_id, text), score=score, rank=rank, retriever="base")


def test_name_identifies_the_cross_encoder_model() -> None:
    reranker = CrossEncoderReranker(model=_StandInCrossEncoder())
    assert reranker.name == "cross-encoder:Xenova/ms-marco-MiniLM-L-6-v2"
    renamed = CrossEncoderReranker("my-model", model=_StandInCrossEncoder())
    assert renamed.name == "cross-encoder:my-model"


def test_score_returns_empty_without_loading_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = CrossEncoderReranker()

    def _explode() -> CrossEncoderModel:
        raise AssertionError("the model must not be loaded for an empty text list")

    monkeypatch.setattr(reranker, "_load", _explode)
    assert reranker.score("anything at all", []) == []


def test_score_passes_batch_size_and_returns_model_scores() -> None:
    model = _StandInCrossEncoder()
    scores = CrossEncoderReranker(batch_size=7, model=model).score(
        "alpha beta", ["alpha one", "nothing here", "beta"]
    )
    assert scores == [1.0, 0.0, 1.0]
    assert model.calls == [("alpha beta", ["alpha one", "nothing here", "beta"], 7)]


def test_score_rejects_malformed_model_output() -> None:
    with pytest.raises(RuntimeError, match="1 scores for 2 texts"):
        CrossEncoderReranker(model=_StandInCrossEncoder(scores=[0.5])).score("q", ["a", "b"])
    with pytest.raises(RuntimeError, match="NaN or infinite"):
        CrossEncoderReranker(model=_StandInCrossEncoder(scores=[0.5, float("nan")])).score(
            "q", ["a", "b"]
        )


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        CrossEncoderReranker(batch_size=0)


def test_base_is_asked_for_max_k_candidates_with_same_filters() -> None:
    base = _RecordingBase([_hit("c1", "alpha beta", score=2.5, rank=1)])
    reranker = _FixedReranker({"alpha beta": 1.0})
    retriever = RerankingRetriever(base, reranker, candidates=3)
    filters: Filters = {"source": "doc-1.md"}

    out = retriever.retrieve("alpha", k=2, filters=filters)
    assert base.calls == [("alpha", 3, filters)]
    assert [(hit.chunk.chunk_id, hit.score, hit.rank) for hit in out] == [("c1", 1.0, 1)]
    assert out[0].components == {"base": 2.5, "base_rank": 1.0}

    # k above candidates: the base is asked for k hits, and filters are forwarded again.
    retriever.retrieve("alpha", k=10, filters=filters)
    assert base.calls[-1] == ("alpha", 10, filters)
    assert reranker.calls == [("alpha", ["alpha beta"]), ("alpha", ["alpha beta"])]


def test_reranking_reorders_and_renumbers_lexical_hits() -> None:
    lexical = _lexical_retriever()
    query = "bm25 lexical"
    base_hits = lexical.retrieve(query, 10)
    assert len(base_hits) >= 3
    texts = [hit.chunk.embedding_text for hit in base_hits]
    # Increasing scores per first-stage rank: reranking must exactly reverse the base order.
    scores = {text: float(index) for index, text in enumerate(texts)}
    reranker = _FixedReranker(scores)
    retriever = RerankingRetriever(lexical, reranker, candidates=10)
    assert retriever.name == "lexical+rerank"
    assert retriever.base is lexical
    assert retriever.reranker is reranker

    results = retriever.retrieve(query, k=len(texts), filters=None)
    assert reranker.calls == [(query, texts)]
    assert [hit.chunk.embedding_text for hit in results] == list(reversed(texts))
    assert [hit.score for hit in results] == sorted(scores.values(), reverse=True)
    assert [hit.rank for hit in results] == list(range(1, len(texts) + 1))
    assert {hit.retriever for hit in results} == {"lexical+rerank"}
    rank_by_text = {hit.chunk.embedding_text: hit for hit in base_hits}
    for hit in results:
        base_hit = rank_by_text[hit.chunk.embedding_text]
        assert hit.components["lexical"] == pytest.approx(base_hit.score)
        assert hit.components["base_rank"] == float(base_hit.rank)

    truncated = retriever.retrieve(query, k=2, filters=None)
    assert [hit.chunk.embedding_text for hit in truncated] == list(reversed(texts))[:2]
    assert [hit.rank for hit in truncated] == [1, 2]


def test_ties_keep_first_stage_order() -> None:
    lexical = _lexical_retriever()
    query = "bm25 lexical"
    base_hits = lexical.retrieve(query, 10)
    assert len(base_hits) >= 2
    retriever = RerankingRetriever(lexical, _FixedReranker({}), candidates=10)

    results = retriever.retrieve(query, k=len(base_hits), filters=None)
    assert [hit.chunk.chunk_id for hit in results] == [hit.chunk.chunk_id for hit in base_hits]
    assert [hit.rank for hit in results] == list(range(1, len(base_hits) + 1))
    assert all(hit.score == 0.0 for hit in results)


def test_invalid_k_and_candidates_raise_value_error() -> None:
    base = _RecordingBase([])
    with pytest.raises(ValueError, match="candidates must be >= 1"):
        RerankingRetriever(base, _FixedReranker({}), candidates=0)
    retriever = RerankingRetriever(base, _FixedReranker({}))
    with pytest.raises(ValueError, match="k must be positive"):
        retriever.retrieve("alpha", k=0)
    assert base.calls == []


def test_blank_query_skips_base_and_reranker() -> None:
    base = _RecordingBase([_hit("c1", "alpha", score=1.0, rank=1)])
    reranker = _FixedReranker({})
    retriever = RerankingRetriever(base, reranker)
    assert retriever.retrieve("   ", k=3) == []
    assert base.calls == []
    assert reranker.calls == []


def test_no_first_stage_hits_skips_reranker() -> None:
    base = _RecordingBase([])
    reranker = _FixedReranker({})
    retriever = RerankingRetriever(base, reranker, candidates=4)
    assert retriever.retrieve("alpha", k=2) == []
    assert base.calls == [("alpha", 4, None)]
    assert reranker.calls == []


def test_reranker_score_count_mismatch_raises_runtime_error() -> None:
    hits = [_hit("c1", "alpha", score=1.0, rank=1), _hit("c2", "beta", score=0.5, rank=2)]
    retriever = RerankingRetriever(_RecordingBase(hits), _ShortReranker())
    with pytest.raises(RuntimeError, match="returned 1 scores for 2 candidates"):
        retriever.retrieve("alpha")


@pytest.mark.slow
def test_real_cross_encoder_prefers_the_bm25_passage() -> None:
    pytest.importorskip("fastembed")
    scores = CrossEncoderReranker(threads=2).score(
        "what is bm25",
        [
            "BM25 is a lexical ranking function based on term frequency and document length.",
            "The cat sat on the mat.",
        ],
    )
    assert scores[0] > scores[1]


@pytest.mark.slow
def test_real_reranker_puts_the_joint_scoring_passage_first() -> None:
    pytest.importorskip("fastembed")
    passages = [
        "Bi-encoders embed the query and each passage separately, which is fast.",
        "Cross-encoders score the query and passage jointly, which is slower but more accurate.",
        "BM25 ranks documents by term frequency and document length.",
    ]
    dense = DenseRetriever(HashingEmbedder())
    dense.index([_chunk(f"p{i}", text, doc_id=f"d{i}") for i, text in enumerate(passages)])
    retriever = RerankingRetriever(dense, CrossEncoderReranker(threads=2))

    results = retriever.retrieve("why are cross encoders more accurate than bi-encoders", k=3)
    assert results
    assert results[0].chunk.text == passages[1]
