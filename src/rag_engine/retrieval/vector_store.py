"""Exact in-process vector index (ADR-0001).

Vectors are stored as one contiguous float32 matrix; search is a single
matrix-vector product followed by a partial sort. Metadata filters are
applied as a boolean mask *before* ranking, so filtered queries still return
the true top-k among matching chunks.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rag_engine.models import Chunk
from rag_engine.retrieval.embedders import Matrix, Vector, l2_normalize

FilterValue = str | int | float | bool | Sequence[str | int | float | bool]
Filters = Mapping[str, FilterValue]

STORE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    rank: int  # 1-based


class VectorStore(Protocol):
    @property
    def dimension(self) -> int: ...

    def __len__(self) -> int: ...

    def upsert(self, chunks: Sequence[Chunk], vectors: Matrix) -> None: ...

    def delete_document(self, doc_id: str) -> int: ...

    def search(
        self, query: Vector, k: int = 10, filters: Filters | None = None
    ) -> list[SearchHit]: ...


def _field(chunk: Chunk, key: str) -> Any:
    if key in ("doc_id", "page"):
        return getattr(chunk, key)
    if key == "heading_path":
        return list(chunk.heading_path)
    return chunk.metadata.get(key)


def matches(chunk: Chunk, filters: Filters) -> bool:
    """Equality filters; a list filter means "any of"; list-valued fields match on overlap."""
    for key, expected in filters.items():
        actual = _field(chunk, key)
        is_list = isinstance(expected, Sequence) and not isinstance(expected, str)
        wanted = set(expected) if is_list else {expected}  # type: ignore[arg-type]
        values = set(actual) if isinstance(actual, list) else {actual}
        if not values & wanted:
            return False
    return True


class InMemoryVectorStore:
    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._chunks: list[Chunk] = []
        self._matrix: Matrix = np.zeros((0, dimension), dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    def _validate(self, vectors: Matrix, rows: int) -> Matrix:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape != (rows, self._dimension):
            raise ValueError(
                f"expected vectors of shape ({rows}, {self._dimension}), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("vectors contain NaN or infinite values")
        return l2_normalize(array)

    def upsert(self, chunks: Sequence[Chunk], vectors: Matrix) -> None:
        """Insert chunks; any existing chunks of the same documents are replaced.

        Document-level replacement means re-indexing an edited file never leaves
        stale chunks behind, even when the number of chunks changes.
        """
        normalized = self._validate(vectors, len(chunks))
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk_id in upsert batch")
        doc_ids = {c.doc_id for c in chunks}
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id not in doc_ids]
        self._chunks = [self._chunks[i] for i in keep] + list(chunks)
        self._matrix = np.vstack([self._matrix[keep], normalized]).astype(np.float32)

    def delete_document(self, doc_id: str) -> int:
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._chunks = [self._chunks[i] for i in keep]
            self._matrix = self._matrix[keep]
        return removed

    def search(self, query: Vector, k: int = 10, filters: Filters | None = None) -> list[SearchHit]:
        if k <= 0:
            raise ValueError("k must be positive")
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape != (self._dimension,):
            raise ValueError(f"query has dimension {q.shape[0]}, index has {self._dimension}")
        if not self._chunks:
            return []
        norm = float(np.linalg.norm(q))
        q = q / norm if norm > 0 else q
        scores = self._matrix @ q
        candidates = np.arange(len(self._chunks))
        if filters:
            mask = np.fromiter((matches(c, filters) for c in self._chunks), dtype=bool)
            candidates = candidates[mask]
        if candidates.size == 0:
            return []
        k = min(k, candidates.size)
        candidate_scores = scores[candidates]
        if k < candidates.size:
            # O(n) selection of the k-th best score, then keep every candidate tied
            # with it so the tie-break below is exact rather than arbitrary.
            kth = np.partition(-candidate_scores, k - 1)[k - 1]
            selected = np.flatnonzero(-candidate_scores <= kth)
        else:
            selected = np.arange(candidates.size)
        # Order: score descending, then insertion order (deterministic ties).
        order = np.lexsort((candidates[selected], -candidate_scores[selected]))[:k]
        hits: list[SearchHit] = []
        for rank, position in enumerate(selected[order], start=1):
            hits.append(
                SearchHit(
                    chunk=self._chunks[int(candidates[position])],
                    score=float(candidate_scores[position]),
                    rank=rank,
                )
            )
        return hits

    # ------------------------------------------------------------------ persistence

    def save(self, directory: str | Path, embedder_name: str | None = None) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._matrix)
        with (path / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(chunk.model_dump_json() + "\n")
        manifest = {
            "format_version": STORE_FORMAT_VERSION,
            "dimension": self._dimension,
            "count": len(self._chunks),
            "embedder": embedder_name,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> InMemoryVectorStore:
        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != STORE_FORMAT_VERSION:
            raise ValueError(f"unsupported index format {manifest.get('format_version')!r}")
        store = cls(int(manifest["dimension"]))
        matrix = np.load(path / "vectors.npy", allow_pickle=False)
        lines = (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
        if len(chunks) != manifest["count"] or matrix.shape != (len(chunks), store.dimension):
            raise ValueError("index files are inconsistent with manifest")
        store._chunks = chunks
        store._matrix = matrix.astype(np.float32)
        return store
