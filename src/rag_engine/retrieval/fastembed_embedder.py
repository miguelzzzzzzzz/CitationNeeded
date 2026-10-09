"""Dense embeddings from a local ONNX model via fastembed (ADR-0002).

``fastembed`` is an optional dependency (``pip install -e ".[embeddings]"``)
and is imported lazily, so the core package and the default test suite do
not need it. The model is downloaded on first use (~67 MB for the default
``BAAI/bge-small-en-v1.5``, which fastembed serves as a quantized ONNX export)
and cached in ``cache_dir``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import numpy as np

from rag_engine.config import EmbeddingConfig
from rag_engine.retrieval.embedders import Matrix, Vector, l2_normalize


class TextEmbeddingModel(Protocol):
    """The subset of ``fastembed.TextEmbedding`` this adapter uses."""

    def embed(self, documents: Iterable[str], batch_size: int = ...) -> Iterable[Any]: ...

    def query_embed(self, query: Iterable[str]) -> Iterable[Any]: ...


def _model_dimension(model_name: str) -> int | None:
    """Look up the output dimension without loading the model (None if unknown)."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return None

    for description in TextEmbedding.list_supported_models():
        if description.get("model") == model_name:
            dim = description.get("dim")
            return int(dim) if dim is not None else None
    return None


class FastEmbedEmbedder:
    """``Embedder`` backed by a fastembed ``TextEmbedding`` model.

    * Documents are embedded in batches of ``batch_size``. Each batch is
      copied into one preallocated float32 matrix, so peak memory is bounded
      by the batch, not by the corpus, beyond the output itself.
    * Empty or whitespace-only texts map to zero vectors without calling the
      model (the ``Embedder`` contract), so they never match anything.
    * Rows are L2-normalized here rather than trusting the model, so inner
      product equals cosine similarity for every backend.
    * ``query_prefix`` supports instruction-style query prefixes (BGE
      suggests one for short queries); it is empty by default and its effect
      is measured in the evaluation milestone, not assumed.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        model: TextEmbeddingModel | None = None,
        dimension: int | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        self._model = model
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"fastembed:{self.config.model_name}"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = _model_dimension(self.config.model_name)
        if self._dimension is None:
            # Unknown to the registry (custom model): embed a probe once.
            self._dimension = int(self._embed_batch(["dimension probe"], query=False).shape[1])
        return self._dimension

    def _load(self) -> TextEmbeddingModel:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - depends on install
                raise ImportError(
                    "fastembed is not installed; run: pip install -e '.[embeddings]'"
                ) from exc
            self._model = TextEmbedding(
                model_name=self.config.model_name,
                cache_dir=self.config.cache_dir,
                threads=self.config.threads,
            )
        return self._model

    def _embed_batch(self, texts: Sequence[str], *, query: bool) -> Matrix:
        model = self._load()
        if query:
            rows = list(model.query_embed(list(texts)))
        else:
            rows = list(model.embed(list(texts), batch_size=self.config.batch_size))
        if len(rows) != len(texts):
            raise RuntimeError(f"model returned {len(rows)} vectors for {len(texts)} texts")
        matrix = np.asarray(np.stack(rows), dtype=np.float32)
        if matrix.ndim != 2:
            raise RuntimeError(f"model returned vectors of shape {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise RuntimeError("model returned NaN or infinite values")
        return matrix

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        output: Matrix | None = None
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        non_empty = [i for i, text in enumerate(texts) if text.strip()]
        size = self.config.batch_size
        for start in range(0, len(non_empty), size):
            indices = non_empty[start : start + size]
            vectors = self._embed_batch([texts[i] for i in indices], query=False)
            if output is None:
                if self._dimension is not None and vectors.shape[1] != self._dimension:
                    raise RuntimeError(
                        f"model returned dimension {vectors.shape[1]}, expected {self._dimension}"
                    )
                self._dimension = int(vectors.shape[1])
                output = np.zeros((len(texts), self._dimension), dtype=np.float32)
            elif vectors.shape[1] != output.shape[1]:
                raise RuntimeError("model returned inconsistent dimensions across batches")
            output[indices] = vectors
        if output is None:  # every text was empty
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        return l2_normalize(output)

    def embed_query(self, text: str) -> Vector:
        if not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)
        matrix = self._embed_batch([self.config.query_prefix + text], query=True)
        vector: Vector = l2_normalize(matrix)[0]
        return vector
