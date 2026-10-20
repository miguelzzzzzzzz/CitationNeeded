"""Request and response models of the HTTP API (no FastAPI import).

Requests reject unknown fields and enforce the same option rules as
``rag-engine search``/``answer``, so a combination the CLI refuses (e.g.
``fusion`` outside hybrid mode) is a 422 here rather than an option that is
silently ignored. Hits carry the same fields as ``search --json`` hits.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_engine.answering.select import GeneratorChoice
from rag_engine.retrieval.compose import RetrievalMode
from rag_engine.retrieval.retriever import RetrievedChunk

__all__ = [
    "MAX_FILTER_KEYS",
    "MAX_FILTER_VALUES",
    "MAX_QUERY_CHARS",
    "SEARCH_SCHEMA_VERSION",
    "AnswerRequest",
    "HealthResponse",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
]

# Same payload version as ``rag-engine search --json``; bump on breaking changes.
SEARCH_SCHEMA_VERSION = 1
MAX_QUERY_CHARS = 4096
MAX_FILTER_KEYS = 16
MAX_FILTER_VALUES = 100

FilterScalar = str | int | float | bool
FilterValue = FilterScalar | list[FilterScalar]

_STRICT = ConfigDict(frozen=True, extra="forbid")


class RetrievalOptions(BaseModel):
    """Retrieval options shared by ``POST /search`` and ``POST /answer``.

    ``None`` means "not given": the defaults (``rrf`` fusion, ``rrf_k`` 60, a
    pool of 50 candidates) are applied by the ``effective_*`` helpers, so the
    validator can tell an explicit option from an omitted one.
    """

    model_config = _STRICT

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    k: int = Field(default=5, ge=1, le=100)
    mode: RetrievalMode = "hybrid"
    fusion: Literal["rrf", "weighted"] | None = None
    rrf_k: int | None = Field(default=None, ge=0)
    weights: dict[Literal["dense", "lexical"], float] | None = None
    candidates: int | None = Field(default=None, ge=1, le=1000)
    rerank: bool = False
    filters: dict[str, FilterValue] | None = Field(default=None, max_length=MAX_FILTER_KEYS)

    @field_validator("query")
    @classmethod
    def _non_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value

    @field_validator("filters")
    @classmethod
    def _valid_filters(cls, value: dict[str, FilterValue] | None) -> dict[str, FilterValue] | None:
        if value is None:
            return None
        for key, expected in value.items():
            if not key:
                raise ValueError("filter keys must be non-empty")
            if isinstance(expected, list):
                if not expected:
                    raise ValueError(f"filter {key!r}: an 'any of' list must not be empty")
                if len(expected) > MAX_FILTER_VALUES:
                    raise ValueError(f"filter {key!r}: at most {MAX_FILTER_VALUES} values")
                values: list[FilterScalar] = expected
            else:
                values = [expected]
            for item in values:
                if isinstance(item, float) and not math.isfinite(item):
                    raise ValueError(f"filter {key!r}: values must be finite")
        return value

    @model_validator(mode="after")
    def _consistent_options(self) -> Self:
        if self.mode != "hybrid" and any(
            option is not None for option in (self.fusion, self.rrf_k, self.weights)
        ):
            raise ValueError("fusion, rrf_k and weights require mode 'hybrid'")
        if self.rrf_k is not None and self.fusion == "weighted":
            raise ValueError("rrf_k requires fusion 'rrf'")
        if self.candidates is not None and self.mode != "hybrid" and not self.rerank:
            # The pool size would be passed to a retriever that ignores it.
            raise ValueError("candidates requires mode 'hybrid' or rerank")
        if self.weights is not None:
            if not self.weights:
                raise ValueError("weights must name 'dense' and/or 'lexical'")
            for name, weight in self.weights.items():
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError(f"weight for {name!r} must be finite and >= 0")
            if all(weight == 0 for weight in self.weights.values()):
                raise ValueError("at least one weight must be positive")
        return self

    def effective_fusion(self) -> str:
        return self.fusion or "rrf"

    def effective_rrf_k(self) -> int:
        return 60 if self.rrf_k is None else self.rrf_k

    def effective_candidates(self) -> int:
        return 50 if self.candidates is None else self.candidates


class SearchRequest(RetrievalOptions):
    """Body of ``POST /search``."""


class AnswerRequest(RetrievalOptions):
    """Body of ``POST /answer``; ``k`` is the number of chunks given to the generator."""

    max_sentences: int = Field(default=5, ge=1, le=20)
    generator: GeneratorChoice = "auto"


class SearchHit(BaseModel):
    """One ranked chunk with its citation provenance (offsets index normalized text).

    ``components``/``ranks`` hold the per-stage scores and ranks after fusion or
    reranking and are empty for a single retriever.
    """

    model_config = _STRICT

    rank: int
    score: float
    retriever: str
    components: dict[str, float] = Field(default_factory=dict)
    ranks: dict[str, int] = Field(default_factory=dict)
    doc_id: str
    chunk_id: str
    source: str | None
    content_hash: str | None
    heading_path: list[str]
    page: int | None
    page_end: int | None
    start_char: int
    end_char: int
    text: str

    @classmethod
    def from_hit(cls, hit: RetrievedChunk) -> SearchHit:
        payload: dict[str, Any] = hit.to_dict(include_text=True)
        return cls.model_validate(payload)


class SearchResponse(BaseModel):
    """Body returned by ``POST /search``."""

    model_config = _STRICT

    schema_version: int = SEARCH_SCHEMA_VERSION
    query: str
    mode: RetrievalMode
    rerank: bool
    hits: list[SearchHit]


class HealthResponse(BaseModel):
    """Body of ``GET /health``: 200 with ``ok`` when an index is served, else 503."""

    model_config = _STRICT

    status: Literal["ok", "unavailable"]
    version: str
    index_loaded: bool
    embedder: str | None = None
    chunks: int = 0
    documents: int = 0
    detail: str | None = None
