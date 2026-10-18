"""Tests for ``rag_engine.eval.intervals`` and ``scripts/answer_eval_table.py``."""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import pytest

from rag_engine.eval.intervals import wilson_interval

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "answer_eval_table.py"

_M_PARSE = "JSON replies parsed"
_M_SPAN = "citation span validity (quote verbatim in cited context, >= 20 chars)"
_M_RECALL = "citation recall vs gold docs in context (answerable, gold retrieved)"
_M_PRECISION = "citation precision (valid citations to a gold doc)"
_M_FALSE_ABSTENTION = "false abstention (answerable, gold in context)"
_M_CORRECT_ABSTENTION = "correct abstention (gold removed; lower bound)"

_MISMATCH = "records give 1.0 but metrics say 0.5"


@pytest.fixture(scope="module")
def answer_eval_table() -> Any:
    """Load the reporting script as a module without importing it as a package."""
    spec = importlib.util.spec_from_file_location("answer_eval_table", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their defining module through ``sys.modules``, so the
    # entry must exist before ``exec_module`` runs.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _grade(
    *,
    kind: str = "answerable",
    parse_ok: bool = True,
    abstained: bool = False,
    n_raw_citations: int = 1,
    n_valid_citations: int = 1,
    n_valid_gold_citations: int = 1,
    gold_doc_ids_in_context: list[str] | None = None,
    citation_recall: float | None = 1.0,
) -> dict[str, Any]:
    gold = ["d1"] if gold_doc_ids_in_context is None else gold_doc_ids_in_context
    return {
        "query_id": "q-1",
        "kind": kind,
        "parse_ok": parse_ok,
        "abstained": abstained,
        "n_raw_citations": n_raw_citations,
        "n_valid_citations": n_valid_citations,
        "n_valid_gold_citations": n_valid_gold_citations,
        "final_citations_consistent": True,
        "gold_doc_ids_in_context": gold,
        "cited_doc_ids": ["d1"],
        "citation_recall": citation_recall,
        "citation_precision": 1.0,
        "abstention_correct": True,
        "error": None,
    }


def _report(grades: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a report whose ``metrics`` are recomputed like the script expects."""
    n_raw = sum(grade["n_raw_citations"] for grade in grades)
    n_valid = sum(grade["n_valid_citations"] for grade in grades)
    n_valid_gold = sum(grade["n_valid_gold_citations"] for grade in grades)
    recalls = [grade["citation_recall"] for grade in grades if grade["citation_recall"] is not None]
    answerable_with_gold = [
        grade
        for grade in grades
        if grade["kind"] == "answerable" and grade["gold_doc_ids_in_context"]
    ]
    no_relevant = [grade for grade in grades if grade["kind"] == "no_relevant"]
    parse_ok = sum(1 for grade in grades if grade["parse_ok"])
    false_abstentions = sum(1 for grade in answerable_with_gold if grade["abstained"])
    correct_abstentions = sum(1 for grade in no_relevant if grade["abstained"])
    false_abstention_rate = (
        false_abstentions / len(answerable_with_gold) if answerable_with_gold else None
    )
    correct_abstention_rate = correct_abstentions / len(no_relevant) if no_relevant else None
    metrics: dict[str, Any] = {
        "parse_ok_rate": parse_ok / len(grades) if grades else None,
        "citation_span_validity": n_valid / n_raw if n_raw else None,
        "citation_recall": statistics.fmean(recalls) if recalls else None,
        "citation_precision": n_valid_gold / n_valid if n_valid else None,
        "false_abstention_rate": false_abstention_rate,
        "correct_abstention_rate": correct_abstention_rate,
    }
    return {
        "records": [{"grade": grade} for grade in grades],
        "metrics": metrics,
        "config": {"model": "test-model"},
        "git_sha": "deadbeef",
    }


def _ci(successes: int, n: int, *, confidence: float = 0.95) -> str:
    low, high = wilson_interval(successes, n, confidence=confidence)
    return f"[{low:.2f}, {high:.2f}]"


def _write_report(tmp_path: Path, report: dict[str, Any], name: str = "report.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("successes", "n", "expected"),
    [
        (8, 10, (0.4902, 0.9433)),
        (0, 10, (0.0, 0.2775)),
        (10, 10, (0.7225, 1.0)),
        (29, 30, (0.8333, 0.9941)),
    ],
)
def test_wilson_interval_matches_hand_computed_values(
    successes: int, n: int, expected: tuple[float, float]
) -> None:
    low, high = wilson_interval(successes, n, confidence=0.95)
    assert low == pytest.approx(expected[0], abs=1e-4)
    assert high == pytest.approx(expected[1], abs=1e-4)


def test_wilson_interval_saturates_at_unit_bounds() -> None:
    low_zero, high_zero = wilson_interval(0, 10)
    assert low_zero == 0.0
    assert 0.0 < high_zero < 1.0

    low_full, high_full = wilson_interval(10, 10)
    assert high_full == 1.0
    assert 0.0 < low_full < 1.0


def test_wilson_interval_widens_with_confidence() -> None:
    low_95, high_95 = wilson_interval(8, 10, confidence=0.95)
    low_99, high_99 = wilson_interval(8, 10, confidence=0.99)
    assert low_99 < low_95 < high_95 < high_99


@pytest.mark.parametrize(
    ("successes", "n", "confidence", "error"),
    [
        pytest.param(0, 0, 0.95, ValueError, id="zero-trials"),
        pytest.param(11, 10, 0.95, ValueError, id="too-many-successes"),
        pytest.param(-1, 10, 0.95, ValueError, id="negative-successes"),
        pytest.param(5, 10, 0.0, ValueError, id="zero-confidence"),
        pytest.param(5, 10, 1.0, ValueError, id="unit-confidence"),
        pytest.param(True, 10, 0.95, TypeError, id="bool-successes"),
        pytest.param(5, True, 0.95, TypeError, id="bool-trials"),
        pytest.param(5, 10.0, 0.95, TypeError, id="float-trials"),
        pytest.param(5, 10, True, TypeError, id="bool-confidence"),
        pytest.param(5, 10, "0.95", TypeError, id="str-confidence"),
    ],
)
def test_wilson_interval_rejects_invalid_arguments(
    successes: Any, n: Any, confidence: Any, error: type[Exception]
) -> None:
    with pytest.raises(error):
        wilson_interval(successes, n, confidence=confidence)


def test_build_rows_binary_report_gives_all_six_rows(answer_eval_table: Any) -> None:
    grades = [
        _grade(
            parse_ok=True,
            abstained=False,
            n_raw_citations=2,
            n_valid_citations=2,
            n_valid_gold_citations=1,
            citation_recall=1.0,
        ),
        _grade(
            parse_ok=True,
            abstained=True,
            n_raw_citations=2,
            n_valid_citations=1,
            n_valid_gold_citations=1,
            gold_doc_ids_in_context=["d2"],
            citation_recall=0.0,
        ),
        _grade(
            parse_ok=False,
            abstained=False,
            n_raw_citations=0,
            n_valid_citations=0,
            n_valid_gold_citations=0,
            gold_doc_ids_in_context=["d3"],
            citation_recall=None,
        ),
        _grade(
            kind="no_relevant",
            parse_ok=True,
            abstained=True,
            n_raw_citations=1,
            n_valid_citations=1,
            n_valid_gold_citations=0,
            gold_doc_ids_in_context=[],
            citation_recall=None,
        ),
    ]

    rows = answer_eval_table.build_rows(_report(grades), confidence=0.95)

    assert [(row.metric, row.value, row.count, row.ci) for row in rows] == [
        (_M_PARSE, "0.75", "3 / 4", _ci(3, 4)),
        (_M_SPAN, "0.80", "4 / 5", _ci(4, 5)),
        (_M_RECALL, "0.50", "1 / 2", _ci(1, 2)),
        (_M_PRECISION, "0.50", "2 / 4", _ci(2, 4)),
        (_M_FALSE_ABSTENTION, "0.33", "1 / 3", _ci(1, 3)),
        (_M_CORRECT_ABSTENTION, "1.00", "1 / 1", _ci(1, 1)),
    ]


def test_build_rows_non_binary_recall_uses_mean(answer_eval_table: Any) -> None:
    grades = [_grade(citation_recall=1.0), _grade(citation_recall=0.5)]

    rows = answer_eval_table.build_rows(_report(grades), confidence=0.95)
    recall = next(row for row in rows if row.metric == _M_RECALL)

    assert (recall.value, recall.count, recall.ci) == (
        "0.75",
        "mean over 2 items",
        "n/a (not binary)",
    )


def test_build_rows_zero_denominator_correct_abstention_is_not_applicable(
    answer_eval_table: Any,
) -> None:
    rows = answer_eval_table.build_rows(_report([_grade(citation_recall=1.0)]), confidence=0.95)
    row = next(row for row in rows if row.metric == _M_CORRECT_ABSTENTION)

    assert (row.value, row.count, row.ci) == ("n/a", "0 / 0", "n/a")


def test_build_rows_recall_without_values_is_not_applicable(answer_eval_table: Any) -> None:
    grades = [_grade(citation_recall=None)]
    rows = answer_eval_table.build_rows(_report(grades), confidence=0.95)
    row = next(row for row in rows if row.metric == _M_RECALL)

    assert (row.value, row.count, row.ci) == ("n/a", "0 / 0", "n/a")


def test_build_rows_without_records_marks_every_row_not_applicable(
    answer_eval_table: Any,
) -> None:
    rows = answer_eval_table.build_rows(_report([]), confidence=0.95)

    assert len(rows) == 6
    assert all((row.value, row.count, row.ci) == ("n/a", "0 / 0", "n/a") for row in rows)


def test_build_rows_rejects_metric_drift(answer_eval_table: Any) -> None:
    report = _report([_grade(citation_recall=1.0)])
    report["metrics"]["parse_ok_rate"] = 0.5

    with pytest.raises(answer_eval_table.ReportMismatchError, match=_MISMATCH):
        answer_eval_table.build_rows(report, confidence=0.95)


@pytest.mark.parametrize(
    ("metric", "bad_value"),
    [
        ("parse_ok_rate", "high"),
        ("parse_ok_rate", True),
        ("citation_recall", None),
    ],
)
def test_build_rows_rejects_non_numeric_metrics(
    answer_eval_table: Any, metric: str, bad_value: object
) -> None:
    report = _report([_grade(citation_recall=1.0)])
    report["metrics"][metric] = bad_value

    with pytest.raises(answer_eval_table.ReportMismatchError, match="expected a number"):
        answer_eval_table.build_rows(report, confidence=0.95)


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("n_raw_citations", "2", "grade.n_raw_citations must be an int"),
        ("citation_recall", "1.0", "grade.citation_recall must be a number"),
    ],
)
def test_build_rows_rejects_malformed_grade_fields(
    answer_eval_table: Any, field: str, bad_value: str, match: str
) -> None:
    report = _report([_grade(citation_recall=1.0)])
    report["records"][0]["grade"][field] = bad_value

    with pytest.raises(answer_eval_table.ReportMismatchError, match=match):
        answer_eval_table.build_rows(report, confidence=0.95)


def test_render_table_contains_header_separator_and_rows(answer_eval_table: Any) -> None:
    rows = answer_eval_table.build_rows(_report([_grade(citation_recall=1.0)]), confidence=0.95)

    table = answer_eval_table.render_table(rows, confidence=0.95)

    assert "| metric | value | count | 95% Wilson CI |" in table
    assert "| --- | --- | --- | --- |" in table
    assert table.count("\n") == len(rows) + 1


def test_render_table_uses_requested_confidence(answer_eval_table: Any) -> None:
    rows = answer_eval_table.build_rows(_report([_grade(citation_recall=1.0)]), confidence=0.99)

    table = answer_eval_table.render_table(rows, confidence=0.99)

    assert "| metric | value | count | 99% Wilson CI |" in table


def test_main_prints_table_and_source_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], answer_eval_table: Any
) -> None:
    path = _write_report(tmp_path, _report([_grade(citation_recall=1.0)]))

    assert answer_eval_table.main([str(path)]) == 0

    out = capsys.readouterr().out
    assert "| metric | value | count | 95% Wilson CI |" in out
    assert f"Source: {path}, model test-model, git_sha deadbeef." in out


def test_main_honours_confidence_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], answer_eval_table: Any
) -> None:
    path = _write_report(tmp_path, _report([_grade(citation_recall=1.0)]))

    assert answer_eval_table.main([str(path), "--confidence", "0.99"]) == 0

    assert "| metric | value | count | 99% Wilson CI |" in capsys.readouterr().out


def test_main_returns_non_zero_on_metric_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], answer_eval_table: Any
) -> None:
    report = _report([_grade(citation_recall=1.0)])
    report["metrics"]["parse_ok_rate"] = 0.5
    path = _write_report(tmp_path, report)

    assert answer_eval_table.main([str(path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "records give" in captured.err


def test_main_rejects_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], answer_eval_table: Any
) -> None:
    assert answer_eval_table.main([str(tmp_path / "missing.json")]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")


def test_main_rejects_invalid_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], answer_eval_table: Any
) -> None:
    path = tmp_path / "report.json"
    path.write_text("{not json", encoding="utf-8")

    assert answer_eval_table.main([str(path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")


def test_main_rejects_out_of_range_confidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], answer_eval_table: Any
) -> None:
    path = _write_report(tmp_path, _report([_grade(citation_recall=1.0)]))

    assert answer_eval_table.main([str(path), "--confidence", "1.5"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("[]", id="report-not-an-object"),
        pytest.param('{"metrics": {}}', id="records-missing"),
        pytest.param('{"records": {}}', id="records-not-a-list"),
        pytest.param('{"records": [1]}', id="record-not-an-object"),
        pytest.param('{"records": [{"grade": 1}]}', id="grade-not-an-object"),
        pytest.param('{"records": []}', id="metrics-missing"),
    ],
)
def test_main_rejects_malformed_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], answer_eval_table: Any, payload: str
) -> None:
    path = tmp_path / "report.json"
    path.write_text(payload, encoding="utf-8")

    assert answer_eval_table.main([str(path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
