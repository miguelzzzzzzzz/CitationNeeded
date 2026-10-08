"""Embedding interface and a deterministic, dependency-free implementation.

Real models (fastembed / ONNX, ADR-0002) implement the same ``Embedder``
protocol. ``HashingEmbedder`` exists so the full index/search path can be
tested quickly and offline, and doubles as a cheap lexical baseline.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
Matrix = NDArray[np.float32]

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Maps text to fixed-size float32 vectors.

    Implementations must return L2-normalized rows (or all-zero rows for
    inputs with no content) so inner product equals cosine similarity.
    """

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Vector: ...


def l2_normalize(matrix: Matrix) -> Matrix:
    """Row-normalize; rows with zero norm are left as zeros."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return (matrix / safe).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing of word unigrams and bigrams with sublinear TF.

    Deterministic across processes and platforms (uses BLAKE2b, not
    Python's randomized ``hash``).
    """

    def __init__(self, dimension: int = 512, use_bigrams: bool = True) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        self._dimension = dimension
        self._use_bigrams = use_bigrams

    @property
    def name(self) -> str:
        return f"hashing-{self._dimension}{'-bigrams' if self._use_bigrams else ''}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _features(self, text: str) -> list[str]:
        words = _WORD.findall(text.lower())
        features = list(words)
        if self._use_bigrams:
            features += [f"{a} {b}" for a, b in pairwise(words)]
        return features

    def _embed(self, text: str) -> Vector:
        counts: dict[int, float] = {}
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self._dimension
            sign = 1.0 if (value >> 63) & 1 else -1.0
            counts[index] = counts.get(index, 0.0) + sign
        vector = np.zeros(self._dimension, dtype=np.float32)
        for index, count in counts.items():
            vector[index] = np.sign(count) * (1.0 + np.log(abs(count))) if count else 0.0
        return vector

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return l2_normalize(np.stack([self._embed(t) for t in texts]))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector
