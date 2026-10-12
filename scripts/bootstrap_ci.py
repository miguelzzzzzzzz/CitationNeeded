#!/usr/bin/env python3
"""CLI for paired bootstrap confidence intervals on retrieval mode scores."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rag_engine.eval.bootstrap import load_per_query_scores, paired_bootstrap_ci


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Paired bootstrap CI for two retrieval modes on aligned per-query scores.")
    )
    parser.add_argument("path", type=Path, help="Path to .per_query.json sidecar")
    parser.add_argument("--mode-a", default="hybrid", help="First mode (default: hybrid)")
    parser.add_argument("--mode-b", default="dense", help="Second mode (default: dense)")
    parser.add_argument("--metric", default="ndcg@10", help="Metric name (default: ndcg@10)")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Bootstrap samples (default: 10000)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit strict JSON instead of human text",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        scores_a, scores_b, _ = load_per_query_scores(
            args.path,
            mode_a=args.mode_a,
            mode_b=args.mode_b,
            metric=args.metric,
        )
        ci = paired_bootstrap_ci(
            scores_a,
            scores_b,
            n_bootstrap=args.n_bootstrap,
            confidence=args.confidence,
            seed=args.seed,
        )
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = asdict(ci)
        payload.update(
            mode_a=args.mode_a,
            mode_b=args.mode_b,
            metric=args.metric,
        )
        print(json.dumps(payload, allow_nan=False))
        return 0

    confidence_pct = round(ci.confidence * 100)
    print(f"paired bootstrap CI: {args.mode_a} - {args.mode_b} on {args.metric}")
    print(
        f"n={ci.n}  mean_diff={ci.mean_diff:.6f}  "
        f"{confidence_pct}% CI=[{ci.ci_low:.6f}, {ci.ci_high:.6f}]  "
        f"(B={ci.n_bootstrap}, seed={ci.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
