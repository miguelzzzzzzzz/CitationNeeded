"""Descriptive statistics for a set of chunks (used by the CLI and evals)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag_engine.models import Chunk


def chunk_stats(chunks: Sequence[Chunk]) -> dict[str, float | int]:
    if not chunks:
        return {"chunks": 0, "documents": 0, "total_tokens": 0}
    counts = [c.token_count for c in chunks]
    sizes = np.array(counts, dtype=np.float64)
    return {
        "chunks": len(chunks),
        "documents": len({c.doc_id for c in chunks}),
        "total_tokens": sum(counts),
        "tokens_min": min(counts),
        "tokens_mean": round(float(sizes.mean()), 2),
        "tokens_p50": round(float(np.percentile(sizes, 50)), 2),
        "tokens_p95": round(float(np.percentile(sizes, 95)), 2),
        "tokens_max": max(counts),
    }
