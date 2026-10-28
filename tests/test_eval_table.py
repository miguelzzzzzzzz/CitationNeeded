"""Tests for ``scripts/eval_table.py`` (README tables come only from its output)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_table.py"
_COMMITTED = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "results"
    / "scifact_test_fastembed_20261012T013527Z.json"
)


@pytest.fixture(scope="module")
def eval_table() -> Any:
    spec = importlib.util.spec_from_file_location("eval_table", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise
    return module


def _report() -> dict[str, Any]:
    scores = {
        "recall@1": 0.25,
        "hit@1": 0.5,
        "recall@5": 0.75,
        "hit@5": 1.0,
        "recall@10": 1.0,
        "hit@10": 1.0,
        "mrr@10": 0.625,
        "ndcg@10": 0.7,
    }
    return {
        "git_sha": "0123456789abcdef",
        "git_dirty": False,
        "n_queries": 4,
        "hardware": {"platform": "Linux-x86_64", "processor": "", "cpu_count": 2},
        "metrics": {"dense": scores, "lexical": {**scores, "ndcg@10": 0.5}},
        "latency_ms": {"dense": {"retrieve_p50": 12.345, "retrieve_p95": 99.96}},
    }


def test_render_report_has_at_1_and_at_5_columns(eval_table: Any) -> None:
    out = eval_table.render_report(_report(), "r.json")
    lines = out.splitlines()
    assert lines[0] == (
        "| mode | nDCG@10 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | Hit@1 | Hit@5 | Hit@10 |"
    )
    assert (
        "| dense | 0.7000 | 0.6250 | 0.2500 | 0.7500 | 1.0000 | 0.5000 | 1.0000 | 1.0000 |" in lines
    )
    assert "| lexical | 0.5000 |" in out


def test_render_report_latency_per_mode_and_missing_mode(eval_table: Any) -> None:
    out = eval_table.render_report(_report(), "r.json")
    assert "| dense | 12.3 | 100.0 |" in out
    # No timings recorded for lexical: shown as n/a, never invented.
    assert "| lexical | n/a | n/a |" in out


def test_render_report_states_hardware_verbatim(eval_table: Any) -> None:
    out = eval_table.render_report(_report(), "r.json")
    assert (
        "Hardware as recorded in the report: platform `Linux-x86_64`, processor (empty), "
        "cpu_count `2`." in out
    )
    assert "Source: `r.json` (n_queries 4, git_sha `0123456`, git_dirty false)." in out


def test_render_report_without_hardware_or_sha(eval_table: Any) -> None:
    report = _report()
    del report["hardware"], report["git_sha"], report["git_dirty"], report["latency_ms"]
    report["metrics"]["broken"] = "not a dict"
    out = eval_table.render_report(report, "r.json")
    assert "Hardware as recorded in the report: not recorded." in out
    assert "git_sha `?`, git_dirty null" in out
    assert "| broken |" not in out.split("\n\n")[0]


def test_fmt_ms_rejects_non_numbers(eval_table: Any) -> None:
    assert eval_table._fmt_ms(True) == "n/a"
    assert eval_table._fmt_ms(None) == "n/a"
    assert eval_table._fmt_ms(3) == "3.0"


def test_main_report_flag_renders_committed_report(
    eval_table: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert eval_table.main(["--report", str(_COMMITTED)]) == 0
    out = capsys.readouterr().out
    payload = json.loads(_COMMITTED.read_text(encoding="utf-8"))
    p95 = payload["latency_ms"]["hybrid"]["retrieve_p95"]
    assert f"| hybrid | {payload['latency_ms']['hybrid']['retrieve_p50']:.1f} | {p95:.1f} |" in out
    assert payload["hardware"]["platform"] in out


def test_main_report_flag_errors(
    eval_table: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert eval_table.main(["--report", str(tmp_path / "missing.json")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert eval_table.main(["--report", str(bad)]) == 2
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"rows": []}), encoding="utf-8")
    assert eval_table.main(["--report", str(other)]) == 2
    assert "not an eval report" in capsys.readouterr().err


def test_main_summary_mode_still_works(
    eval_table: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a.json").write_text(json.dumps(_report()), encoding="utf-8")
    assert eval_table.main([str(tmp_path)]) == 0
    assert "| a.json | ? | ? | dense | 0.7000 |" in capsys.readouterr().out
