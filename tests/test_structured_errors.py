from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.eval.structured import load_structured_docs


def _write_root(tmp_path: Path, qrels_line: str) -> Path:
    root = tmp_path / "root"
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "a.md").write_text("# A\n\n## One\n\nalpha text\n", encoding="utf-8")
    (root / "qrels.jsonl").write_text(qrels_line + "\n", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("qrels_line", "fragment"),
    [
        (
            json.dumps([1, 2]),
            "expected a JSON object",
        ),
        (
            json.dumps({"text": "t", "relevant_docs": ["a"]}),
            "'query_id' must be a non-empty string",
        ),
        (
            json.dumps({"query_id": "q", "text": "t", "relevant_docs": "a"}),
            "'relevant_docs' must be a list of file stems",
        ),
        (
            json.dumps({"query_id": "q", "text": "t", "relevant_docs": [""]}),
            "entries must be non-empty strings",
        ),
        (
            json.dumps(
                {
                    "query_id": "q",
                    "text": "t",
                    "relevant_docs": ["a"],
                    "relevant_sections": "One",
                }
            ),
            "'relevant_sections' must be a list of heading paths",
        ),
        (
            json.dumps(
                {
                    "query_id": "q",
                    "text": "t",
                    "relevant_docs": ["a"],
                    "relevant_sections": ["One"],
                }
            ),
            "each 'relevant_sections' entry must be a list of headings",
        ),
        (
            json.dumps(
                {
                    "query_id": "q",
                    "text": "t",
                    "relevant_docs": ["a"],
                    "relevant_sections": [["One"], ["Two"]],
                }
            ),
            "'relevant_sections' has 2 entries",
        ),
        (
            "not json",
            "invalid JSON",
        ),
        (
            json.dumps({"query_id": "q", "text": "t", "relevant_docs": ["zzz"]}),
            "unknown document stems",
        ),
    ],
)
def test_load_structured_docs_rejects_malformed_qrels(
    tmp_path: Path,
    qrels_line: str,
    fragment: str,
) -> None:
    root = _write_root(tmp_path, qrels_line)

    with pytest.raises(ValueError) as excinfo:
        load_structured_docs(root)

    assert fragment in str(excinfo.value)
