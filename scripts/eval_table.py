#!/usr/bin/env python3
"""Render a Markdown comparison table from committed evals/results/*.json reports.

Usage:
  python scripts/eval_table.py [results_dir]

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
        if path.name.endswith(".tmp.json"):
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path("evals/results"),
        help="Directory of JSON reports (default: evals/results).",
    )
    args = parser.parse_args(argv)
    if not args.results_dir.is_dir():
        print(f"error: not a directory: {args.results_dir}", file=sys.stderr)
        return 2
    sys.stdout.write(render(_load_reports(args.results_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
