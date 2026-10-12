"""Paired bootstrap confidence intervals for retrieval mode comparison.

Percentile CI uses the inclusive nearest-rank method: sort the bootstrap means
and take indices floor(p * (B - 1)) and ceil(p * (B - 1)).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BootstrapCI:
    """Result of a paired bootstrap confidence-interval calculation."""

    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    confidence: float
    n_bootstrap: int
    seed: int
    n_positive: int
    n_negative: int


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Compute a paired bootstrap confidence interval for mean(scores_a - scores_b)."""
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length; "
            f"got {len(scores_a)} and {len(scores_b)}"
        )
    n = len(scores_a)
    if n < 1:
        raise ValueError("scores_a and scores_b must contain at least one observation")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1; got {n_bootstrap}")

    diffs = [float(a) - float(b) for a, b in zip(scores_a, scores_b, strict=True)]
    mean_diff = sum(diffs) / n

    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(n_bootstrap):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        bootstrap_means.append(total / n)

    bootstrap_means.sort()
    p_low = (1.0 - confidence) / 2.0
    p_high = 1.0 - p_low
    low_idx = math.floor(p_low * (n_bootstrap - 1))
    high_idx = math.ceil(p_high * (n_bootstrap - 1))
    low_idx = max(0, min(low_idx, n_bootstrap - 1))
    high_idx = max(0, min(high_idx, n_bootstrap - 1))

    n_positive = sum(1 for m in bootstrap_means if m > 0.0)
    n_negative = sum(1 for m in bootstrap_means if m < 0.0)

    return BootstrapCI(
        n=n,
        mean_diff=mean_diff,
        ci_low=bootstrap_means[low_idx],
        ci_high=bootstrap_means[high_idx],
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_positive=n_positive,
        n_negative=n_negative,
    )


def load_per_query_scores(
    path: Path,
    *,
    mode_a: str,
    mode_b: str,
    metric: str,
) -> tuple[list[float], list[float], list[str]]:
    """Load aligned per-query scores for two modes from a sidecar JSON file."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    schema_version = data.get("schema_version")
    if schema_version is not None and schema_version != 1:
        raise ValueError(f"unsupported schema_version {schema_version!r}; expected 1")

    modes = data.get("modes")
    if not isinstance(modes, dict):
        raise ValueError("sidecar JSON is missing a 'modes' object")

    for mode in (mode_a, mode_b):
        if mode not in modes:
            raise ValueError(f"mode {mode!r} not found in sidecar modes: {sorted(modes)}")
        if not isinstance(modes[mode], dict):
            raise ValueError(f"mode {mode!r} does not contain a query map")

    if "query_ids" in data:
        query_ids_raw = data["query_ids"]
        if not isinstance(query_ids_raw, list):
            raise ValueError("'query_ids' must be a list")
        query_ids = [str(q) for q in query_ids_raw]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("'query_ids' contains duplicate entries")
    else:
        ids_a = {str(q) for q in modes[mode_a]}
        ids_b = {str(q) for q in modes[mode_b]}
        if ids_a != ids_b:
            raise ValueError(
                "mode query id sets differ; provide explicit 'query_ids' to align them"
            )
        query_ids = sorted(ids_a)

    scores_a: list[float] = []
    scores_b: list[float] = []
    for qid in query_ids:
        for mode, scores in ((mode_a, scores_a), (mode_b, scores_b)):
            mode_data = modes[mode]
            if qid not in mode_data:
                raise ValueError(f"query id {qid!r} missing from mode {mode!r}")
            query_data = mode_data[qid]
            if not isinstance(query_data, dict):
                raise ValueError(f"mode {mode!r} query {qid!r} does not contain a metric map")
            if metric not in query_data:
                raise ValueError(f"metric {metric!r} missing for mode {mode!r} query {qid!r}")
            value = query_data[metric]
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"metric {metric!r} for mode {mode!r} query {qid!r} is not numeric: {value!r}"
                )
            scores.append(float(value))

    return scores_a, scores_b, query_ids
