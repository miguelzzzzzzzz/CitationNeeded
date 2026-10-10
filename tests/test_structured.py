"""Tests for :mod:`rag_engine.eval.structured` against the in-repo eval set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.eval.structured import load_structured_docs

REPO_SET = Path(__file__).resolve().parents[1] / "evals" / "structured"


def test_load_repo_structured_set() -> None:
    docset = load_structured_docs(REPO_SET)
    assert set(docset.documents) == {"metabolism", "climate_policy", "sorting_algorithms"}
    assert len(docset.queries) == 8
    headings = {
        part
        for doc in docset.documents.values()
        for section in doc.sections
        for part in section.heading_path
    }
    assert "Glycolysis" in headings
    qrels = docset.qrels_binary()
    assert set(qrels) == set(docset.queries)
    metabolism_id = docset.documents["metabolism"].doc_id
    assert qrels["q_glycolysis_atp"] == {metabolism_id: 1}
    # stem map matches documents
    assert docset.stem_to_doc_id()["metabolism"] == metabolism_id


def test_load_structured_missing_stem(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "only.md").write_text("# Only\n\nbody\n", encoding="utf-8")
    (tmp_path / "qrels.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q",
                "text": "t",
                "relevant_docs": ["missing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing"):
        load_structured_docs(tmp_path)


def test_load_structured_requires_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="docs"):
        load_structured_docs(tmp_path)


def test_duplicate_query_id_rejected(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "only.md").write_text("# Only\n\nbody\n", encoding="utf-8")
    line = json.dumps({"query_id": "q", "text": "t", "relevant_docs": ["only"]})
    (tmp_path / "qrels.jsonl").write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_structured_docs(tmp_path)


def test_missing_qrels_file(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A\n\nx\n", encoding="utf-8")
    with pytest.raises(ValueError, match="qrels"):
        load_structured_docs(tmp_path)


def test_invalid_qrels_shapes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "only.md").write_text("# Only\n\nbody\n", encoding="utf-8")
    (tmp_path / "qrels.jsonl").write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_structured_docs(tmp_path)


def test_relevant_sections_length_mismatch(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "only.md").write_text("# Only\n\n## Sec\n\nbody\n", encoding="utf-8")
    (tmp_path / "qrels.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q",
                "text": "t",
                "relevant_docs": ["only"],
                "relevant_sections": [["Sec"], ["Extra"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="relevant_sections"):
        load_structured_docs(tmp_path)


def test_non_object_qrels_line(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "only.md").write_text("# Only\n\nbody\n", encoding="utf-8")
    (tmp_path / "qrels.jsonl").write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_structured_docs(tmp_path)
