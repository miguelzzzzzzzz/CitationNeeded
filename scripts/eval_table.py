#!/usr/bin/env python3
"""Render a Markdown comparison table from committed evals/results/*.json reports.

Usage:
  python scripts/eval_table.py [results_dir]        # one summary row per report and mode
  python scripts/eval_table.py --report REPORT.json # one report: @1/@5/@10, latency, hardware

Only reads reports; never invents metrics. README tables must be copied from
this output (or an equivalent committed report), never typed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_reports(results_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith(".tmp.json") or path.name.endswith(".per_query.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skip {path.name}: {exc}", file=sys.stderr)
            continue
        if not isinstance(payload, dict) or "metrics" not in payload:
            print(f"warning: skip {path.name}: not an eval report", file=sys.stderr)
            continue
        payload["_path"] = path.name
        reports.append(payload)
    return reports


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return "_No committed evaluation reports under evals/results/ yet._\n"
    lines = [
        "| report | dataset | embedder | mode | nDCG@10 | MRR@10 | Recall@10 | Hit@10 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        config = report.get("config") or {}
        embedder = config.get("embedder", "?")
        dataset = report.get("dataset_name", "?")
        metrics = report.get("metrics") or {}
        for mode, scores in metrics.items():
            if not isinstance(scores, dict):
                continue
            row = (
                f"| {report['_path']} | {dataset} | {embedder} | {mode} | "
                f"{_fmt(scores.get('ndcg@10', ''))} | {_fmt(scores.get('mrr@10', ''))} | "
                f"{_fmt(scores.get('recall@10', ''))} | {_fmt(scores.get('hit@10', ''))} |"
            )
            lines.append(row)
    lines.append("")
    lines.append("Source: committed JSON under `evals/results/`. Do not edit numbers by hand.")
    lines.append("")
    return "\n".join(lines)


_QUALITY_COLUMNS = (
    ("nDCG@10", "ndcg@10"),
    ("MRR@10", "mrr@10"),
    ("Recall@1", "recall@1"),
    ("Recall@5", "recall@5"),
    ("Recall@10", "recall@10"),
    ("Hit@1", "hit@1"),
    ("Hit@5", "hit@5"),
    ("Hit@10", "hit@10"),
)


def _fmt_ms(value: object) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{value:.1f}"
    return "n/a"


def _fmt_hardware(hardware: object) -> str:
    """The report's hardware record, verbatim; empty values are shown as such."""
    if not isinstance(hardware, dict) or not hardware:
        return "not recorded"
    parts = []
    for key, value in hardware.items():
        shown = "(empty)" if value in ("", None) else f"`{value}`"
        parts.append(f"{key} {shown}")
    return ", ".join(parts)


def render_report(report: dict[str, Any], name: str) -> str:
    """Quality (@1/@5/@10) and per-mode latency tables for a single report."""
    metrics = report.get("metrics") or {}
    latency = report.get("latency_ms") or {}
    lines = [
        "| mode | " + " | ".join(label for label, _ in _QUALITY_COLUMNS) + " |",
        "| --- |" + " ---: |" * len(_QUALITY_COLUMNS),
    ]
    for mode, scores in metrics.items():
        if not isinstance(scores, dict):
            continue
        cells = " | ".join(_fmt(scores.get(key, "")) for _, key in _QUALITY_COLUMNS)
        lines.append(f"| {mode} | {cells} |")
    lines += [
        "",
        "| mode | retrieve p50 (ms) | retrieve p95 (ms) |",
        "| --- | ---: | ---: |",
    ]
    for mode in metrics:
        timings = latency.get(mode) or {}
        lines.append(
            f"| {mode} | {_fmt_ms(timings.get('retrieve_p50'))} | "
            f"{_fmt_ms(timings.get('retrieve_p95'))} |"
        )
    lines += [
        "",
        f"Source: `{name}` (n_queries {report.get('n_queries', '?')}, "
        f"git_sha `{str(report.get('git_sha') or '?')[:7]}`, "
        f"git_dirty {json.dumps(report.get('git_dirty'))}).",
        f"Hardware as recorded in the report: {_fmt_hardware(report.get('hardware'))}.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path("evals/results"),
        help="Directory of JSON reports (default: evals/results).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Render one report in detail (@1/@5/@10 metrics, per-mode latency, hardware).",
    )
    args = parser.parse_args(argv)
    if args.report is not None:
        try:
            payload = json.loads(args.report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot read {args.report}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(payload, dict) or "metrics" not in payload:
            print(f"error: not an eval report: {args.report}", file=sys.stderr)
            return 2
        sys.stdout.write(render_report(payload, args.report.name))
        return 0
    if not args.results_dir.is_dir():
        print(f"error: not a directory: {args.results_dir}", file=sys.stderr)
        return 2
    sys.stdout.write(render(_load_reports(args.results_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
