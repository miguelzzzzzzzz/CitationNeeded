"""End-to-end citation-offset invariant tests for the fusion and reranking stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import pytest

from rag_engine.chunking import chunk_documents
from rag_engine.config import ChunkingConfig, ChunkStrategy, IngestionConfig
from rag_engine.documents import verify_chunks
from rag_engine.ingestion import ingest_path
from rag_engine.models import Chunk, Document
from rag_engine.retrieval.embedders import HashingEmbedder
from rag_engine.retrieval.fusion import HybridRetriever
from rag_engine.retrieval.rerank import RerankingRetriever
from rag_engine.retrieval.retriever import DenseRetriever, LexicalRetriever, RetrievedChunk

_INTRO_MD = (
    "---\ntitle: Intro Guide\ntags: rag\n---\n"
    "# Intro\r\n"
    "\r\n"
    "The \ufb01rst \ufb02ow explains retrieval and citations.\r\n"
    "\r\n"
    "## Setup\r\n"
    "\r\n"
    "Install the package, then build the index.\r\n"
)

_API_HTML = (
    "<html><head><title>API</title></head><body><h1>API</h1>"
    "<p>Search returns cited chunks.</p><h2>Ranking</h2>"
    "<p>Fusion combines dense and lexical retrieval.</p></body></html>"
)

_NOTES_TXT = (
    "Notes on BM25\x07 and dense\x0b vectors.\n"
    "\uff32\uff45\uff54\uff52\uff49\uff45\uff56\uff41\uff4c quality matters.\n"
)

_QUERIES: tuple[str, ...] = (
    "retrieval",
    "citations",
    "fusion",
    "ranking",
    "dense vectors",
    "BM25",
)


class _OverlapReranker:
    """Offline reranker scoring by the number of shared lowercase words."""

    @property
    def name(self) -> str:
        return "overlap"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        query_words = set(query.lower().split())
        return [float(len(query_words & set(text.lower().split()))) for text in texts]


class _E2E(NamedTuple):
    docs_by_id: Mapping[str, Document]
    chunks: Sequence[Chunk]
    dense: DenseRetriever
    lexical: LexicalRetriever
    hybrid: HybridRetriever


def assert_offsets(hits: Sequence[RetrievedChunk], docs_by_id: Mapping[str, Document]) -> None:
    for hit in hits:
        document = docs_by_id[hit.chunk.doc_id]
        assert document.text[hit.chunk.start_char : hit.chunk.end_char] == hit.chunk.text
        provenance = hit.provenance()
        assert provenance["doc_id"] == hit.chunk.doc_id
        assert provenance["chunk_id"] == hit.chunk.chunk_id
        assert provenance["content_hash"] == document.content_hash


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory) -> _E2E:
    kb = tmp_path_factory.mktemp("e2e") / "kb"
    (kb / "guide").mkdir(parents=True)
    (kb / "ref").mkdir(parents=True)
    (kb / "guide" / "intro.md").write_text(_INTRO_MD, encoding="utf-8", newline="")
    (kb / "ref" / "api.html").write_bytes(_API_HTML.encode("utf-8"))
    (kb / "notes.txt").write_text(_NOTES_TXT, encoding="utf-8", newline="")

    report = ingest_path(kb, IngestionConfig(corpus_id="e2e"))
    documents = list(report.documents)
    docs_by_id = {document.doc_id: document for document in documents}

    chunks = chunk_documents(
        documents,
        ChunkingConfig(
            strategy=ChunkStrategy.RECURSIVE, chunk_size=12, chunk_overlap=3, min_chunk_tokens=1
        ),
    )
    verify_chunks(chunks, documents)
    assert chunks

    dense = DenseRetriever(HashingEmbedder(dimension=64, use_bigrams=False))
    lexical = LexicalRetriever()
    dense.index(chunks)
    lexical.index(chunks)
    hybrid = HybridRetriever([dense, lexical], method="rrf")
    return _E2E(
        docs_by_id=docs_by_id,
        chunks=chunks,
        dense=dense,
        lexical=lexical,
        hybrid=hybrid,
    )


def test_rrf_fusion_offsets_and_stage_keys(e2e: _E2E) -> None:
    rrf = HybridRetriever([e2e.dense, e2e.lexical], method="rrf")
    total = 0
    for query in _QUERIES:
        hits = rrf.retrieve(query, k=5)
        assert_offsets(hits, e2e.docs_by_id)
        for hit in hits:
            assert hit.ranks
            assert hit.components
            assert set(hit.ranks) <= {"dense", "lexical"}
            assert set(hit.components) <= {"dense", "lexical"}
        total += len(hits)
    assert total > 0


def test_weighted_fusion_offsets(e2e: _E2E) -> None:
    weighted = HybridRetriever(
        [e2e.dense, e2e.lexical], method="weighted", weights={"lexical": 2.0}
    )
    assert weighted.name
    total = 0
    for query in _QUERIES:
        hits = weighted.retrieve(query, k=5)
        assert_offsets(hits, e2e.docs_by_id)
        for hit in hits:
            assert hit.components
        total += len(hits)
    assert total > 0


def test_reranking_offsets_and_stage_keys(e2e: _E2E) -> None:
    hybrid = e2e.hybrid
    reranked = RerankingRetriever(hybrid, _OverlapReranker(), candidates=10)
    assert reranked.name == f"{hybrid.name}+rerank"

    total = 0
    for query in _QUERIES:
        hits = reranked.retrieve(query, k=5)
        assert_offsets(hits, e2e.docs_by_id)
        for hit in hits:
            assert hybrid.name in hit.ranks
            assert set(hit.ranks) <= {"dense", "lexical", hybrid.name}
            assert hybrid.name in hit.components
            assert set(hit.components) <= {"dense", "lexical", hybrid.name}
        total += len(hits)
    assert total > 0


def test_nested_rerank_preserves_inner_ranks(e2e: _E2E) -> None:
    hybrid = e2e.hybrid
    reranker = _OverlapReranker()
    single = RerankingRetriever(hybrid, reranker, candidates=10)
    nested = RerankingRetriever(single, reranker, candidates=10)
    stage_key = single.name
    assert stage_key == f"{hybrid.name}+rerank"
    assert nested.name == f"{single.name}+rerank"

    total = 0
    preserved = 0
    for query in _QUERIES:
        outer_hits = nested.retrieve(query, k=10)
        inner_by_chunk = {hit.chunk.chunk_id: hit for hit in single.retrieve(query, k=10)}
        assert_offsets(outer_hits, e2e.docs_by_id)
        for hit in outer_hits:
            assert hybrid.name in hit.ranks
            assert stage_key in hit.ranks
            assert set(hit.ranks) <= {"dense", "lexical", hybrid.name, stage_key}
            inner = inner_by_chunk.get(hit.chunk.chunk_id)
            if inner is not None:
                assert hit.ranks[hybrid.name] == inner.ranks[hybrid.name]
                for key in ("dense", "lexical"):
                    if key in inner.ranks and key in hit.ranks:
                        assert hit.ranks[key] == inner.ranks[key]
                preserved += 1
        total += len(outer_hits)
    assert total > 0
    assert preserved > 0


def test_reranked_hit_to_dict_payload(e2e: _E2E) -> None:
    reranked = RerankingRetriever(e2e.hybrid, _OverlapReranker(), candidates=10)
    payloads = 0
    for query in _QUERIES:
        for hit in reranked.retrieve(query, k=5):
            payload = hit.to_dict()
            assert "ranks" in payload
            assert "components" in payload
            ranks = payload["ranks"]
            assert isinstance(ranks, dict)
            assert set(ranks) == set(hit.ranks)

            document = e2e.docs_by_id[hit.chunk.doc_id]
            start = payload["start_char"]
            end = payload["end_char"]
            text = payload["text"]
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert isinstance(text, str)
            assert document.text[start:end] == text
            assert text == hit.chunk.text
            payloads += 1
    assert payloads > 0
