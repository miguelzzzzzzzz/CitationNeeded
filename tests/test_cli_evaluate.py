"""CLI tests for ``rag-engine evaluate``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.cli import main

REPO_STRUCTURED = Path(__file__).resolve().parents[1] / "evals" / "structured"


def test_evaluate_structured(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = tmp_path / "results"
    code = main(
        [
            "evaluate",
            "--dataset",
            "structured",
            "--data-dir",
            str(REPO_STRUCTURED),
            "--results-dir",
            str(results),
            "--mode",
            "lexical",
            "--embedder",
            "hashing",
            "--corpus-id",
            "structured-eval",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_documents"] == 3
    assert out["n_queries"] == 8
    assert "lexical" in out["metrics"]
    assert list(results.glob("structured_*.json"))


def test_evaluate_missing_structured(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "evaluate",
            "--dataset",
            "structured",
            "--data-dir",
            str(tmp_path / "nope"),
            "--results-dir",
            str(tmp_path / "out"),
            "--mode",
            "lexical",
        ]
    )
    assert code == 2
    assert "error:" in capsys.readouterr().err
