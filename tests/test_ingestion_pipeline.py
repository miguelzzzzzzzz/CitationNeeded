from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag_engine.config import IngestionConfig
from rag_engine.ingestion import ingest_path


def _reasons(report: object) -> dict[str, str]:
    return {s.source: s.reason for s in report.skipped}  # type: ignore[attr-defined]


def test_ingests_all_supported_formats_in_sorted_order(corpus_dir: Path) -> None:
    report = ingest_path(corpus_dir)
    assert [d.source for d in report.documents] == [
        "notes.txt",
        "reranking.html",
        "retrieval-guide.md",
    ]
    assert {d.format for d in report.documents} == {"text", "html", "markdown"}
    assert report.skipped == []


def test_failures_are_isolated_and_reported(tmp_path: Path, corpus_dir: Path) -> None:
    shutil.copytree(corpus_dir, tmp_path / "docs")
    root = tmp_path / "docs"
    (root / "image.png").write_bytes(b"\x89PNG")
    (root / "empty.md").write_text("\n\n")
    (root / "copy-of-notes.txt").write_text((root / "notes.txt").read_text())
    (root / "big.txt").write_text("word " * 500)
    (root / "sub").mkdir()
    (root / "sub" / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "secret.md").write_text("# should be ignored")

    report = ingest_path(root, IngestionConfig(max_file_bytes=1000))
    reasons = _reasons(report)

    assert "unsupported extension" in reasons["image.png"]
    assert reasons["empty.md"] == "no extractable text"
    assert reasons["notes.txt"] == "duplicate content of copy-of-notes.txt"
    assert reasons["big.txt"].startswith("file too large")
    assert reasons["sub/broken.pdf"].startswith("unreadable PDF")
    assert not any("hidden" in source for source in reasons)
    assert {d.source for d in report.documents} == {
        "copy-of-notes.txt",
        "reranking.html",
        "retrieval-guide.md",
    }
    assert report.files_seen == len(report.documents) + len(report.skipped) == 8


def test_single_file_root_uses_file_name_as_source(corpus_dir: Path) -> None:
    report = ingest_path(corpus_dir / "notes.txt")
    assert [d.source for d in report.documents] == ["notes.txt"]


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest_path(tmp_path / "nope")
