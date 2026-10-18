#!/usr/bin/env python3
"""Render the answer-eval summary table with Wilson score intervals.

Every percentage is recomputed from ``records[*].grade`` rather than trusted
from the aggregate ``metrics`` block, and each recomputed rate is checked
against the reported value so the published table cannot silently drift from
the committed report.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_engine.eval.intervals import wilson_interval

_TOLERANCE = 1e-12

_METRIC_PARSE = "JSON replies parsed"
_METRIC_SPAN = "citation span validity (quote verbatim in cited context, >= 20 chars)"
_METRIC_RECALL = "citation recall vs gold docs in context (answerable, gold retrieved)"
_METRIC_PRECISION = "citation precision (valid citations to a gold doc)"
_METRIC_FALSE_ABSTENTION = "false abstention (answerable, gold in context)"
_METRIC_CORRECT_ABSTENTION = "correct abstention (gold removed; lower bound)"


class ReportMismatchError(RuntimeError):
    """Raised when ``records`` disagree with the report's aggregate metrics."""


@dataclass(frozen=True, slots=True)
class Row:
    """One rendered table row."""

    metric: str
    value: str
    count: str
    ci: str


def _mapping(value: object, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportMismatchError(f"{where} must be a JSON object")
    return value


def _grades(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = report.get("records")
    if not isinstance(records, list):
        raise ReportMismatchError("report must contain a 'records' list")
    grades: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        entry = _mapping(record, where=f"records[{index}]")
        grades.append(_mapping(entry.get("grade"), where=f"records[{index}].grade"))
    return grades


def _as_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportMismatchError(f"grade.{field} must be an int, got {value!r}")
    return value


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReportMismatchError(f"grade.{field} must be a number, got {value!r}")
    return float(value)


def _check_rate(metric: str, recomputed: float, expected: object) -> None:
    if isinstance(expected, bool) or not isinstance(expected, int | float):
        raise ReportMismatchError(
            f"{metric}: report metrics value is {expected!r}, expected a number"
        )
    if abs(recomputed - float(expected)) > _TOLERANCE:
        raise ReportMismatchError(
            f"{metric}: records give {recomputed!r} but metrics say {expected!r}"
        )


def _binary_row(
    metric: str,
    successes: int,
    n: int,
    expected: object,
    *,
    confidence: float,
) -> Row:
    if n == 0:
        return Row(metric=metric, value="n/a", count="0 / 0", ci="n/a")
    rate = successes / n
    _check_rate(metric, rate, expected)
    low, high = wilson_interval(successes, n, confidence=confidence)
    return Row(
        metric=metric,
        value=f"{rate:.2f}",
        count=f"{successes} / {n}",
        ci=f"[{low:.2f}, {high:.2f}]",
    )


def _recall_row(
    grades: list[Mapping[str, Any]],
    expected: object,
    *,
    confidence: float,
) -> Row:
    recalls = [
        _as_float(grade.get("citation_recall"), field="citation_recall")
        for grade in grades
        if grade.get("citation_recall") is not None
    ]
    if not recalls:
        return Row(metric=_METRIC_RECALL, value="n/a", count="0 / 0", ci="n/a")
    if all(value in (0.0, 1.0) for value in recalls):
        successes = sum(1 for value in recalls if value == 1.0)
        return _binary_row(_METRIC_RECALL, successes, len(recalls), expected, confidence=confidence)
    mean = statistics.fmean(recalls)
    _check_rate(_METRIC_RECALL, mean, expected)
    return Row(
        metric=_METRIC_RECALL,
        value=f"{mean:.2f}",
        count=f"mean over {len(recalls)} items",
        ci="n/a (not binary)",
    )


def build_rows(report: Mapping[str, Any], *, confidence: float) -> list[Row]:
    """Recompute every answer-eval rate from ``records`` and build table rows.

    Raises:
        ReportMismatchError: If a recomputed rate disagrees with the
            corresponding ``metrics`` value, or the report is malformed.
    """
    grades = _grades(report)
    metrics = _mapping(report.get("metrics"), where="metrics")

    n_raw = sum(_as_int(grade.get("n_raw_citations"), field="n_raw_citations") for grade in grades)
    n_valid = sum(
        _as_int(grade.get("n_valid_citations"), field="n_valid_citations") for grade in grades
    )
    n_valid_gold = sum(
        _as_int(grade.get("n_valid_gold_citations"), field="n_valid_gold_citations")
        for grade in grades
    )

    answerable_with_gold = [
        grade
        for grade in grades
        if grade.get("kind") == "answerable" and grade.get("gold_doc_ids_in_context")
    ]
    no_relevant = [grade for grade in grades if grade.get("kind") == "no_relevant"]

    return [
        _binary_row(
            _METRIC_PARSE,
            sum(1 for grade in grades if grade.get("parse_ok")),
            len(grades),
            metrics.get("parse_ok_rate"),
            confidence=confidence,
        ),
        _binary_row(
            _METRIC_SPAN,
            n_valid,
            n_raw,
            metrics.get("citation_span_validity"),
            confidence=confidence,
        ),
        _recall_row(grades, metrics.get("citation_recall"), confidence=confidence),
        _binary_row(
            _METRIC_PRECISION,
            n_valid_gold,
            n_valid,
            metrics.get("citation_precision"),
            confidence=confidence,
        ),
        _binary_row(
            _METRIC_FALSE_ABSTENTION,
            sum(1 for grade in answerable_with_gold if grade.get("abstained")),
            len(answerable_with_gold),
            metrics.get("false_abstention_rate"),
            confidence=confidence,
        ),
        _binary_row(
            _METRIC_CORRECT_ABSTENTION,
            sum(1 for grade in no_relevant if grade.get("abstained")),
            len(no_relevant),
            metrics.get("correct_abstention_rate"),
            confidence=confidence,
        ),
    ]


def render_table(rows: list[Row], *, confidence: float) -> str:
    """Render ``rows`` as a GitHub-markdown table with a confidence header."""
    percent = round(confidence * 100)
    lines = [
        f"| metric | value | count | {percent}% Wilson CI |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(f"| {row.metric} | {row.value} | {row.count} | {row.ci} |" for row in rows)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the answer-eval summary table with Wilson score intervals."
    )
    parser.add_argument("path", type=Path, help="Path to the answer-eval report JSON")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for the Wilson intervals (default: 0.95)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        raw: object = json.loads(args.path.read_text(encoding="utf-8"))
        report = _mapping(raw, where="report")
        rows = build_rows(report, confidence=args.confidence)
    except (ReportMismatchError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = report.get("config")
    model = config.get("model") if isinstance(config, Mapping) else None
    print(render_table(rows, confidence=args.confidence))
    print(f"Source: {args.path}, model {model}, git_sha {report.get('git_sha')}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
