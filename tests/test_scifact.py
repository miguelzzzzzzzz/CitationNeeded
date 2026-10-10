"""Tests for :mod:`rag_engine.eval.scifact` using the in-repo mini fixture.

Never hits the network: download paths are exercised with a ``file://`` URL of
``tests/fixtures/eval/scifact_mini.zip``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag_engine.eval.scifact import (
    download_scifact,
    ensure_scifact,
    load_scifact,
    md5_file,
)

FIX = Path(__file__).parent / "fixtures" / "eval"
MINI = FIX / "scifact_mini"
MINI_ZIP = FIX / "scifact_mini.zip"
MINI_MD5 = (FIX / "scifact_mini.md5").read_text(encoding="utf-8").strip()


def test_md5_file_matches_fixture_digest() -> None:
    assert md5_file(MINI_ZIP) == MINI_MD5


def test_load_scifact_test_split() -> None:
    split = load_scifact(MINI, split="test", checksum=MINI_MD5)
    assert split.name == "test"
    assert split.checksum == MINI_MD5
    assert set(split.corpus) == {"doc1", "doc2", "doc3"}
    assert split.corpus["doc1"].title == "Alpha Paper"
    assert "glucose" in split.corpus["doc1"].text
    # q3 has no qrels row, so it is dropped from the evaluation query set
    assert set(split.queries) == {"q1", "q2"}
    assert split.queries["q1"].text.startswith("Does treatment X")
    assert split.qrels == {
        "q1": {"doc3": 1, "doc2": 0},
        "q2": {"doc1": 1},
    }


def test_load_scifact_train_split() -> None:
    split = load_scifact(MINI, split="train")
    assert set(split.queries) == {"q2"}
    assert split.qrels == {"q2": {"doc1": 1}}


def test_load_scifact_unknown_split() -> None:
    with pytest.raises(ValueError, match="unknown SciFact split"):
        load_scifact(MINI, split="dev")


def test_load_scifact_missing_corpus(tmp_path: Path) -> None:
    broken = tmp_path / "scifact"
    shutil.copytree(MINI, broken)
    (broken / "corpus.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match=r"corpus\.jsonl"):
        load_scifact(broken, split="test")


def test_download_scifact_from_file_uri(tmp_path: Path) -> None:
    url = MINI_ZIP.resolve().as_uri()
    root = download_scifact(tmp_path, url=url, expected_md5=MINI_MD5)
    assert root == tmp_path / "scifact"
    assert (root / "corpus.jsonl").is_file()
    assert md5_file(tmp_path / "scifact.zip") == MINI_MD5
    # Second call with a matching cache is a no-op
    again = download_scifact(tmp_path, url=url, expected_md5=MINI_MD5)
    assert again == root
    forced = download_scifact(tmp_path, url=url, expected_md5=MINI_MD5, force=True)
    assert forced == root
    assert (root / "corpus.jsonl").is_file()


def test_download_scifact_checksum_mismatch(tmp_path: Path) -> None:
    url = MINI_ZIP.resolve().as_uri()
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_scifact(tmp_path, url=url, expected_md5="0" * 32)


def test_ensure_scifact(tmp_path: Path) -> None:
    url = MINI_ZIP.resolve().as_uri()
    split = ensure_scifact(tmp_path, split="test", url=url, expected_md5=MINI_MD5)
    assert split.checksum == MINI_MD5
    assert set(split.queries) == {"q1", "q2"}
    assert len(split.corpus) == 3


def test_missing_title_and_text_become_empty(tmp_path: Path) -> None:
    root = tmp_path / "scifact"
    root.mkdir()
    (root / "qrels").mkdir()
    (root / "corpus.jsonl").write_text('{"_id": "d"}\n', encoding="utf-8")
    (root / "queries.jsonl").write_text('{"_id": "q"}\n', encoding="utf-8")
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq\td\t1\n", encoding="utf-8"
    )
    split = load_scifact(root, split="test")
    assert split.corpus["d"].title == ""
    assert split.corpus["d"].text == ""
    assert split.queries["q"].text == ""


def test_download_rejects_zip_slip(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    import zipfile

    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "nope")
        zf.writestr("scifact/corpus.jsonl", "{}\n")
    import hashlib

    digest = hashlib.md5(evil.read_bytes(), usedforsecurity=False).hexdigest()
    with pytest.raises(ValueError, match="unsafe zip entry"):
        download_scifact(tmp_path / "out", url=evil.resolve().as_uri(), expected_md5=digest)
