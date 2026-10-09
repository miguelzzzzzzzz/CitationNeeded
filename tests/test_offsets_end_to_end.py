from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from rag_engine.chunking import chunk_documents
from rag_engine.config import ChunkingConfig, ChunkStrategy, IngestionConfig
from rag_engine.documents import verify_chunks
from rag_engine.ingestion import ingest_path
from rag_engine.models import Chunk, Document
from rag_engine.retrieval.embedders import HashingEmbedder
from rag_engine.retrieval.index_store import load_documents, load_index, save_index
from rag_engine.retrieval.retriever import DenseRetriever, LexicalRetriever, RetrievedChunk

_STRATEGIES: tuple[ChunkStrategy, ...] = tuple(ChunkStrategy)

_INTRO_MD = (
    "---\n"
    "title: Intro Guide\n"
    "tags: rag\n"
    "---\n"
    "# Intro\r\n"
    "\r\n"
    "The \ufb01rst \ufb02ow explains retrieval and citations.\r\n"
    "\r\n"
    "## Setup\r\n"
    "\r\n"
    "Install the package, then build the index.\r\n"
)

_API_HTML = (
    "<html><head><title>API</title></head><body>"
    "<h1>API</h1><p>Search returns cited chunks.</p>"
    "<h2>Ranking</h2><p>Fusion combines dense and lexical retrieval.</p>"
    "</body></html>"
)

_NOTES_TXT = (
    "Notes on BM25\x07 and dense\x0b vectors.\n"
    "\uff32\uff45\uff54\uff52\uff49\uff45\uff56\uff41\uff4c quality matters.\n"
)

_QUERIES: tuple[str, ...] = ("retrieval", "fusion dense lexical", "bm25 vectors")


@dataclass(frozen=True)
class E2EState:
    documents: list[Document]
    docs_by_id: dict[str, Document]
    chunks_by_strategy: dict[ChunkStrategy, list[Chunk]]
    loaded_documents: list[Document]
    loaded_docs_by_id: dict[str, Document]
    loaded_dense: DenseRetriever
    loaded_lexical: LexicalRetriever


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory) -> E2EState:
    root = tmp_path_factory.mktemp("e2e")
    kb = root / "kb"
    (kb / "guide").mkdir(parents=True)
    (kb / "ref").mkdir(parents=True)
    (kb / "guide" / "intro.md").write_text(_INTRO_MD, encoding="utf-8", newline="")
    (kb / "ref" / "api.html").write_text(_API_HTML, encoding="utf-8")
    (kb / "notes.txt").write_text(_NOTES_TXT, encoding="utf-8", newline="")

    report = ingest_path(kb, IngestionConfig(corpus_id="e2e"))
    documents = list(report.documents)
    docs_by_id = {document.doc_id: document for document in documents}

    chunks_by_strategy: dict[ChunkStrategy, list[Chunk]] = {}
    for strategy in _STRATEGIES:
        config = ChunkingConfig(
            strategy=strategy, chunk_size=12, chunk_overlap=3, min_chunk_tokens=1
        )
        chunks_by_strategy[strategy] = list(chunk_documents(documents, config))

    index_chunks = chunks_by_strategy[ChunkStrategy.RECURSIVE]
    dense = DenseRetriever(HashingEmbedder(dimension=64, use_bigrams=False))
    lexical = LexicalRetriever()
    dense.index(index_chunks)
    lexical.index(index_chunks)

    save_dir = root / "store"
    save_index(save_dir, dense, lexical, documents)
    loaded_dense, loaded_lexical = load_index(save_dir)
    loaded_documents = list(load_documents(save_dir))
    loaded_docs_by_id = {document.doc_id: document for document in loaded_documents}

    return E2EState(
        documents=documents,
        docs_by_id=docs_by_id,
        chunks_by_strategy=chunks_by_strategy,
        loaded_documents=loaded_documents,
        loaded_docs_by_id=loaded_docs_by_id,
        loaded_dense=loaded_dense,
        loaded_lexical=loaded_lexical,
    )


def assert_offsets(hits: list[RetrievedChunk], docs_by_id: Mapping[str, Document]) -> None:
    for hit in hits:
        document = docs_by_id[hit.chunk.doc_id]
        start = hit.chunk.start_char
        end = hit.chunk.end_char
        assert 0 <= start <= end <= len(document.text)
        assert document.text[start:end] == hit.chunk.text
        assert hit.provenance()["content_hash"] == document.content_hash


def test_ingestion_and_chunking_offsets(e2e: E2EState) -> None:
    assert len(e2e.documents) == 3
    chunked_doc_ids: set[str] = set()
    for strategy in _STRATEGIES:
        chunks = e2e.chunks_by_strategy[strategy]
        assert chunks, f"no chunks produced for strategy {strategy}"
        verify_chunks(chunks, e2e.documents)
        for chunk in chunks:
            document = e2e.docs_by_id[chunk.doc_id]
            assert document.text[chunk.start_char : chunk.end_char] == chunk.text
        chunked_doc_ids.update(chunk.doc_id for chunk in chunks)
    # Every source document (markdown, html, text) produced chunks under every strategy.
    assert chunked_doc_ids == set(e2e.docs_by_id)


def test_normalized_text_and_front_matter(e2e: E2EState) -> None:
    combined = "\n".join(document.text for document in e2e.documents)
    for document in e2e.documents:
        assert "\r" not in document.text
        assert "\x07" not in document.text
    assert "first flow" in combined
    assert "Retrieval" in combined
    assert "\uff32\uff45\uff54\uff52\uff49\uff45\uff56\uff41\uff4c" not in combined
    assert "title: Intro Guide" not in combined


def test_save_load_retrieval_offsets(e2e: E2EState) -> None:
    assert {d.doc_id for d in e2e.loaded_documents} == set(e2e.docs_by_id)
    produced = False
    for query in _QUERIES:
        dense_hits = e2e.loaded_dense.retrieve(query, k=10)
        lexical_hits = e2e.loaded_lexical.retrieve(query, k=10)
        assert_offsets(dense_hits, e2e.loaded_docs_by_id)
        assert_offsets(lexical_hits, e2e.loaded_docs_by_id)
        if dense_hits or lexical_hits:
            produced = True
    assert produced
