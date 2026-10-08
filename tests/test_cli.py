from __future__ import annotations

import json
from pathlib import Path

from rag_engine.cli import main


def test_ingest_writes_jsonl_and_prints_stats(tmp_path: Path, corpus_dir: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "chunks.jsonl"
    assert (
        main(
            [
                "ingest",
                str(corpus_dir),
                "--out",
                str(out),
                "--strategy",
                "structure",
                "--chunk-size",
                "48",
                "--chunk-overlap",
                "8",
            ]
        )
        == 0
    )
    lines = out.read_text().strip().splitlines()
    assert len(lines) >= 3
    first = json.loads(lines[0])
    assert {"chunk_id", "doc_id", "text", "start_char", "end_char", "token_count"} <= first.keys()
    stats = json.loads(capsys.readouterr().out)
    assert stats["documents"] == 3
    assert stats["chunks"] == len(lines)
    assert stats["strategy"] == "structure"
    assert stats["skipped"] == 0


def test_ingest_reports_skipped_files(tmp_path: Path, corpus_dir: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "docs"
    root.mkdir()
    (root / "ok.txt").write_text("enough words for a chunk here")
    (root / "empty.md").write_text("\n")
    (root / "blob.bin").write_bytes(b"\x00\x01")
    out = tmp_path / "out.jsonl"
    assert main(["ingest", str(root), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    stats = json.loads(captured.out)
    assert stats["documents"] == 1
    assert stats["skipped"] == 2
    assert "empty.md" in captured.err and "blob.bin" in captured.err


def test_invalid_overrides_fail_cleanly(tmp_path: Path, corpus_dir: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "out.jsonl"
    args = ["ingest", str(corpus_dir), "--out", str(out), "--chunk-size", "32"]
    assert main([*args, "--chunk-overlap", "32"]) == 2
    assert "invalid chunking options" in capsys.readouterr().err
    assert not out.exists()


def test_missing_path_fails_cleanly(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["ingest", str(tmp_path / "missing"), "--out", str(tmp_path / "o.jsonl")]) == 2
    assert "does not exist" in capsys.readouterr().err
