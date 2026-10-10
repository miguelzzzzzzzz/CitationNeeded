"""Tests for the ``evaluate --dataset scifact`` CLI branch and its error path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag_engine import cli
from rag_engine.eval.runner import run_structured_eval

_STRUCTURED_DIR = Path(__file__).resolve().parents[1] / "evals" / "structured"


def test_scifact_branch_passes_config_and_prints_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = run_structured_eval(_STRUCTURED_DIR, results_dir=tmp_path / "s")
    seen: list[Any] = []

    def fake_run_scifact_eval(_data_dir: Any, *, config: Any, results_dir: Any) -> Any:
        seen.append(config)
        return report

    monkeypatch.setattr(cli, "run_scifact_eval", fake_run_scifact_eval)

    code = cli.main(
        [
            "evaluate",
            "--dataset",
            "scifact",
            "--data-dir",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "r"),
            "--depth",
            "20",
            "--max-depth",
            "40",
        ]
    )

    assert code == 0
    assert len(seen) == 1
    config = seen[0]
    assert config.dataset == "scifact"
    assert config.depth == 20
    assert config.max_depth == 40
    assert config.corpus_id == "scifact"

    payload = json.loads(capsys.readouterr().out)
    assert "smoke_only" in payload
    assert "retrieval" in payload
    assert "git_dirty" in payload
    assert "n_queries_skipped_no_relevant" in payload


def test_scifact_branch_maps_value_error_to_exit_code_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_scifact_eval(_data_dir: Any, *, config: Any, results_dir: Any) -> Any:
        raise ValueError("boom")

    monkeypatch.setattr(cli, "run_scifact_eval", fake_run_scifact_eval)

    code = cli.main(
        [
            "evaluate",
            "--dataset",
            "scifact",
            "--data-dir",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "r"),
            "--depth",
            "20",
            "--max-depth",
            "40",
        ]
    )

    assert code == 2
    assert "error: boom" in capsys.readouterr().err
