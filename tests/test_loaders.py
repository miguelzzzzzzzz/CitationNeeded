from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from rag_engine.ingestion import (
    HTMLLoader,
    LoaderError,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    loader_for,
)
from rag_engine.ingestion.loaders import find_markdown_headings, split_front_matter
from rag_engine.models import Document

PdfFactory = Callable[..., Path]


def _section_texts(doc: Document) -> list[tuple[tuple[str, ...], str]]:
    return [(s.heading_path, doc.text[s.start_char : s.end_char]) for s in doc.sections]


def _assert_sections_tile_text(doc: Document) -> None:
    assert doc.sections[0].start_char == 0 or doc.format == "pdf"
    for prev, nxt in zip(doc.sections, doc.sections[1:], strict=False):
        assert prev.end_char <= nxt.start_char


class TestMarkdown:
    def test_front_matter_becomes_metadata_and_title(self, corpus_dir: Path) -> None:
        doc = MarkdownLoader().load(corpus_dir / "retrieval-guide.md", "retrieval-guide.md")
        assert doc.title == "Retrieval Engineering Guide"
        assert doc.metadata["tags"] == ["rag", "retrieval"]
        assert doc.metadata["version"] == 2
        assert not doc.text.startswith("---")

    def test_heading_paths_follow_nesting_and_ignore_code_fences(self, corpus_dir: Path) -> None:
        doc = MarkdownLoader().load(corpus_dir / "retrieval-guide.md", "retrieval-guide.md")
        paths = [s.heading_path for s in doc.sections]
        assert paths == [
            (),
            ("Retrieval Engineering Guide",),
            ("Retrieval Engineering Guide", "Lexical search"),
            ("Retrieval Engineering Guide", "Lexical search", "Tuning k1 and b"),
            ("Retrieval Engineering Guide", "Hybrid search"),
        ]
        lexical = dict(_section_texts(doc))[("Retrieval Engineering Guide", "Lexical search")]
        assert "# this comment is not a heading" in lexical
        _assert_sections_tile_text(doc)
        assert doc.sections[-1].end_char == len(doc.text)

    def test_title_falls_back_to_first_h1_then_file_stem(self, tmp_path: Path) -> None:
        with_h1 = tmp_path / "a.md"
        with_h1.write_text("## Sub\n\ntext\n\n# Real Title\n\nbody")
        assert MarkdownLoader().load(with_h1, "a.md").title == "Real Title"
        no_heading = tmp_path / "plain-notes.md"
        no_heading.write_text("just text")
        doc = MarkdownLoader().load(no_heading, "plain-notes.md")
        assert doc.title == "plain-notes"
        assert len(doc.sections) == 1

    def test_invalid_front_matter_is_kept_as_text(self) -> None:
        metadata, body = split_front_matter("---\nkey: [unclosed\n---\nbody")
        assert metadata == {}
        assert body.startswith("---")

    def test_closing_heading_hashes_are_stripped(self) -> None:
        headings = find_markdown_headings("## Setup ##\ntext\n####### not a heading\n")
        assert [(h.level, h.title) for h in headings] == [(2, "Setup")]


class TestHTML:
    def test_extracts_title_meta_and_visible_text_only(self, corpus_dir: Path) -> None:
        doc = HTMLLoader().load(corpus_dir / "reranking.html", "reranking.html")
        assert doc.title == "Cross-Encoder Reranking"
        assert doc.metadata["description"] == "Why rerankers improve precision."
        assert doc.metadata["author"] == "Eval Team"
        assert doc.metadata["language"] == "en"
        for hidden in ("tracking", "color: red", "Home", "Copyright footer"):
            assert hidden not in doc.text
        assert "scores the query and passage jointly, which is slower" in doc.text

    def test_headings_become_sections(self, corpus_dir: Path) -> None:
        doc = HTMLLoader().load(corpus_dir / "reranking.html", "reranking.html")
        sections = dict(_section_texts(doc))
        cost = sections[("Cross-Encoder Reranking", "Cost")]
        assert "- Latency grows linearly with candidates." in cost
        assert "- Rerank only the top 50." in cost


class TestText:
    def test_latin1_fallback_is_recorded(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.txt"
        path.write_bytes("caf\xe9 au lait".encode("latin-1"))
        doc = TextLoader().load(path, "legacy.txt")
        assert doc.text == "café au lait"
        assert doc.metadata["encoding"] == "latin-1"

    def test_binary_content_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.txt"
        path.write_bytes(b"\x00\x01\x02binary")
        with pytest.raises(LoaderError, match="binary"):
            TextLoader().load(path, "blob.txt")

    def test_whitespace_only_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("  \n\n\t\n")
        with pytest.raises(LoaderError, match="no extractable text"):
            TextLoader().load(path, "empty.txt")


class TestPDF:
    def test_pages_become_sections_with_page_numbers(self, pdf_factory: PdfFactory) -> None:
        path = pdf_factory(
            [["First page about BM25."], [], ["Third page about rerank-", "ing models."]],
            title="Search Notes",
        )
        doc = PDFLoader().load(path, "doc.pdf")
        assert doc.title == "Search Notes"
        assert [s.page for s in doc.sections] == [1, 3]
        assert doc.metadata["page_count"] == 3
        assert doc.metadata["empty_pages"] == 1
        page3 = doc.text[doc.sections[1].start_char : doc.sections[1].end_char]
        assert page3 == "Third page about reranking models."
        _assert_sections_tile_text(doc)

    def test_pdf_without_text_is_rejected(self, pdf_factory: PdfFactory) -> None:
        path = pdf_factory([[], []])
        with pytest.raises(LoaderError, match="no extractable text"):
            PDFLoader().load(path, "doc.pdf")

    def test_corrupt_pdf_raises_loader_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
        with pytest.raises(LoaderError):
            PDFLoader().load(path, "broken.pdf")


def test_loader_lookup_is_case_insensitive() -> None:
    assert isinstance(loader_for(Path("README.MD")), MarkdownLoader)
    assert isinstance(loader_for(Path("page.htm")), HTMLLoader)
    assert loader_for(Path("image.png")) is None


def test_doc_id_is_stable_and_content_hash_tracks_changes(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("version one")
    first = TextLoader().load(path, "a.txt")
    path.write_text("version two")
    second = TextLoader().load(path, "a.txt")
    assert first.doc_id == second.doc_id
    assert first.content_hash != second.content_hash
