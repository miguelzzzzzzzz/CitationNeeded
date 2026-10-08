from __future__ import annotations

import random
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import pytest

from rag_engine.chunking import (
    FixedTokenChunker,
    RecursiveChunker,
    StructureAwareChunker,
    TokenIndex,
    build_chunker,
    chunk_documents,
    chunk_stats,
    count_tokens,
)
from rag_engine.config import ChunkingConfig, ChunkStrategy
from rag_engine.ingestion import MarkdownLoader, PDFLoader, ingest_path
from rag_engine.models import Chunk, Document

WORDS = (
    "retrieval", "vector", "index", "query", "passage",
    "rank", "fusion", "score", "model", "latency",
)  # fmt: skip


def synthetic_markdown(seed: int, sections: int = 6) -> str:
    """Deterministic multi-section Markdown with sentences of varying length."""
    rng = random.Random(seed)
    parts = ["# Synthetic Handbook\n"]
    for s in range(sections):
        parts.append(f"## Topic {s}\n")
        for _ in range(rng.randint(1, 4)):
            sentences = []
            for _ in range(rng.randint(1, 6)):
                words = [rng.choice(WORDS) for _ in range(rng.randint(3, 25))]
                sentences.append(" ".join(words).capitalize() + ".")
            parts.append(" ".join(sentences) + "\n")
        if s % 2 == 0:
            parts.append(f"### Detail {s}\n\nShort note {s}.\n")
    return "\n".join(parts)


def load_markdown(tmp_path: Path, text: str, name: str = "doc.md") -> Document:
    path = tmp_path / name
    path.write_text(text)
    return MarkdownLoader().load(path, name)


CONFIGS = [
    ChunkingConfig(strategy=strategy, chunk_size=size, chunk_overlap=overlap, min_chunk_tokens=m)
    for strategy in ChunkStrategy
    for size, overlap, m in [(16, 0, 0), (32, 8, 4), (64, 16, 16), (200, 40, 16)]
]


def _assert_core_invariants(document: Document, chunks: list[Chunk], cfg: ChunkingConfig) -> None:
    assert chunks, "non-empty document must produce chunks"
    tokens = TokenIndex(document.text)
    covered = [False] * len(tokens)
    for i, chunk in enumerate(chunks):
        # exact provenance: the citation span reproduces the chunk text
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.text == chunk.text.strip()
        assert chunk.index == i
        assert chunk.doc_id == document.doc_id
        assert chunk.token_count == count_tokens(chunk.text)
        assert chunk.token_count <= cfg.chunk_size
        first = tokens.first_at_or_after(chunk.start_char)
        last = tokens.first_at_or_after(chunk.end_char)
        for t in range(first, last):
            covered[t] = True
    # no text is lost between chunks
    assert all(covered)
    starts = [c.start_char for c in chunks]
    assert starts == sorted(starts)


def _cfg_id(c: ChunkingConfig) -> str:
    return f"{c.strategy}-{c.chunk_size}-{c.chunk_overlap}"


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_invariants_hold_for_all_strategies(tmp_path: Path, cfg: ChunkingConfig, seed: int) -> None:
    document = load_markdown(tmp_path, synthetic_markdown(seed))
    chunks = build_chunker(cfg).chunk(document)
    _assert_core_invariants(document, chunks, cfg)


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
def test_invariants_hold_on_fixture_corpus(corpus_dir: Path, cfg: ChunkingConfig) -> None:
    for document in ingest_path(corpus_dir).documents:
        _assert_core_invariants(document, build_chunker(cfg).chunk(document), cfg)


def test_fixed_windows_overlap_by_exactly_the_configured_tokens(tmp_path: Path) -> None:
    text = " ".join(f"w{i}" for i in range(100))
    document = load_markdown(tmp_path, text)
    cfg = ChunkingConfig(
        strategy=ChunkStrategy.FIXED, chunk_size=30, chunk_overlap=10, min_chunk_tokens=0
    )
    chunks = FixedTokenChunker(cfg).chunk(document)
    windows = [c.text.split() for c in chunks]
    assert windows[0][0] == "w0" and windows[-1][-1] == "w99"
    for prev, nxt in pairwise(windows):
        assert prev[-10:] == nxt[:10]
    assert [len(w) for w in windows] == [30, 30, 30, 30, 20]


def test_recursive_prefers_sentence_boundaries(tmp_path: Path) -> None:
    sentences = [f"Sentence number {i} talks about hybrid retrieval quality." for i in range(30)]
    document = load_markdown(tmp_path, " ".join(sentences))
    cfg = ChunkingConfig(
        strategy=ChunkStrategy.RECURSIVE, chunk_size=40, chunk_overlap=0, min_chunk_tokens=0
    )
    chunks = RecursiveChunker(cfg).chunk(document)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("Sentence number")
        assert chunk.text.endswith("quality.")


def test_recursive_overlap_repeats_trailing_sentences(tmp_path: Path) -> None:
    sentences = [f"Fact {i} is short." for i in range(40)]
    document = load_markdown(tmp_path, " ".join(sentences))
    cfg = ChunkingConfig(
        strategy=ChunkStrategy.RECURSIVE, chunk_size=20, chunk_overlap=8, min_chunk_tokens=0
    )
    chunks = RecursiveChunker(cfg).chunk(document)
    for prev, nxt in pairwise(chunks):
        assert nxt.start_char < prev.end_char, "consecutive chunks should overlap"
        assert nxt.end_char > prev.end_char, "every chunk must add new text"


def test_recursive_falls_back_to_word_windows_without_separators(tmp_path: Path) -> None:
    document = load_markdown(tmp_path, " ".join(["token"] * 95))
    cfg = ChunkingConfig(
        strategy=ChunkStrategy.RECURSIVE, chunk_size=30, chunk_overlap=0, min_chunk_tokens=0
    )
    chunks = RecursiveChunker(cfg).chunk(document)
    assert [c.token_count for c in chunks] == [30, 30, 30, 5]


def test_structure_chunks_stay_inside_sections_and_carry_heading_paths(tmp_path: Path) -> None:
    document = load_markdown(tmp_path, synthetic_markdown(7))
    cfg = ChunkingConfig(
        strategy=ChunkStrategy.STRUCTURE, chunk_size=48, chunk_overlap=8, min_chunk_tokens=0
    )
    chunks = StructureAwareChunker(cfg).chunk(document)
    for chunk in chunks:
        section = document.section_at(chunk.start_char)
        assert section is not None
        assert chunk.end_char <= section.end_char
        assert chunk.heading_path == section.heading_path
    assert any(c.heading_path[-1:] == ("Detail 0",) for c in chunks)


def test_small_sections_are_merged_into_neighbours(tmp_path: Path) -> None:
    text = "# Guide\n\nTiny.\n\n## A\n\n" + "alpha " * 30 + "\n\n## B\n\nOk.\n"
    document = load_markdown(tmp_path, text)
    cfg = ChunkingConfig(
        strategy=ChunkStrategy.STRUCTURE, chunk_size=64, chunk_overlap=0, min_chunk_tokens=8
    )
    chunks = StructureAwareChunker(cfg).chunk(document)
    assert all(c.token_count >= 8 for c in chunks)
    # merged chunks keep only the heading path shared by everything they cover
    assert chunks[0].heading_path == ("Guide",)
    assert chunks[0].text.startswith("# Guide") and chunks[0].text.endswith("Ok.")


def test_small_chunk_is_kept_when_merging_would_exceed_size(tmp_path: Path) -> None:
    text = "# A\n\n" + "alpha " * 60 + "\n\n# B\n\nTiny tail.\n"
    document = load_markdown(tmp_path, text)
    cfg = ChunkingConfig(
        strategy=ChunkStrategy.STRUCTURE, chunk_size=62, chunk_overlap=0, min_chunk_tokens=8
    )
    chunks = StructureAwareChunker(cfg).chunk(document)
    assert chunks[-1].text == "# B\n\nTiny tail."
    assert chunks[-1].heading_path == ("B",)


def test_context_header_and_embedding_text(corpus_dir: Path) -> None:
    [document] = ingest_path(corpus_dir / "retrieval-guide.md").documents
    cfg = ChunkingConfig(chunk_size=32, chunk_overlap=0, min_chunk_tokens=0)
    chunks = StructureAwareChunker(cfg).chunk(document)
    tuning = next(c for c in chunks if c.heading_path[-1:] == ("Tuning k1 and b",))
    assert tuning.context_header == "Retrieval Engineering Guide > Lexical search > Tuning k1 and b"
    assert tuning.embedding_text.startswith(tuning.context_header + "\n\n")
    assert tuning.metadata["tags"] == ["rag", "retrieval"]
    assert tuning.metadata["source"] == "retrieval-guide.md"

    no_header = ChunkingConfig(include_heading_context=False, chunk_size=32, chunk_overlap=0)
    plain = StructureAwareChunker(no_header).chunk(document)
    assert all(c.embedding_text == c.text for c in plain)


def test_pdf_page_metadata_propagates(pdf_factory: Callable[..., Path]) -> None:
    page = [f"Line {i} about evaluation metrics." for i in range(8)]
    path = pdf_factory([page, page, page])
    document = PDFLoader().load(path, "doc.pdf")
    structure = chunk_documents(
        [document], ChunkingConfig(chunk_size=20, chunk_overlap=0, min_chunk_tokens=0)
    )
    assert {c.page for c in structure} == {1, 2, 3}
    assert all("page_end" not in c.metadata for c in structure)
    fixed = chunk_documents(
        [document],
        ChunkingConfig(
            strategy=ChunkStrategy.FIXED, chunk_size=50, chunk_overlap=0, min_chunk_tokens=0
        ),
    )
    assert any(c.metadata.get("page_end") == 2 for c in fixed if c.page == 1)


def test_chunk_ids_are_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    cfg = ChunkingConfig()
    a = build_chunker(cfg).chunk(load_markdown(tmp_path, synthetic_markdown(3)))
    b = build_chunker(cfg).chunk(load_markdown(tmp_path, synthetic_markdown(3)))
    c = build_chunker(cfg).chunk(load_markdown(tmp_path, synthetic_markdown(4)))
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b]
    assert len({x.chunk_id for x in a}) == len(a)
    assert {x.chunk_id for x in a}.isdisjoint({x.chunk_id for x in c})


def test_chunk_stats(tmp_path: Path) -> None:
    assert chunk_stats([]) == {"chunks": 0, "documents": 0, "total_tokens": 0}
    chunks = build_chunker(ChunkingConfig(chunk_size=32, chunk_overlap=0)).chunk(
        load_markdown(tmp_path, synthetic_markdown(5))
    )
    stats = chunk_stats(chunks)
    assert stats["chunks"] == len(chunks)
    assert stats["documents"] == 1
    assert stats["total_tokens"] == sum(c.token_count for c in chunks)
    assert stats["tokens_max"] <= 32
