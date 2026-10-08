"""Runtime configuration.

Settings are plain pydantic models so they can be validated, serialized into
evaluation reports, and constructed explicitly in tests. ``Settings.from_env``
reads ``RAG_``-prefixed environment variables; nothing else in the package reads
the environment directly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ENV_PREFIX = "RAG_"


class ChunkStrategy(StrEnum):
    """Available chunking strategies (see ``rag_engine.chunking``)."""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    STRUCTURE = "structure"


class ChunkingConfig(BaseModel):
    """Parameters shared by all chunkers.

    Sizes are measured in whitespace-delimited tokens (see
    ``rag_engine.chunking.tokens``), which keeps chunking deterministic and
    independent of any particular model tokenizer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: ChunkStrategy = ChunkStrategy.STRUCTURE
    chunk_size: int = Field(default=256, ge=8, le=8192)
    chunk_overlap: int = Field(default=32, ge=0)
    min_chunk_tokens: int = Field(default=16, ge=0)
    include_heading_context: bool = True

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_tokens > self.chunk_size:
            raise ValueError("min_chunk_tokens must not exceed chunk_size")
        return self


class IngestionConfig(BaseModel):
    """Limits applied while loading source files."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_file_bytes: int = Field(default=20_000_000, gt=0)
    follow_symlinks: bool = False


class Settings(BaseModel):
    """Top-level settings object passed through the pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ingestion: IngestionConfig = IngestionConfig()
    chunking: ChunkingConfig = ChunkingConfig()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``RAG_*`` variables, falling back to defaults.

        Raises ``ValueError`` with a readable message for invalid values.
        """
        env = os.environ if environ is None else environ
        ingestion: dict[str, str] = {}
        chunking: dict[str, str] = {}
        mapping = {
            "MAX_FILE_BYTES": (ingestion, "max_file_bytes"),
            "FOLLOW_SYMLINKS": (ingestion, "follow_symlinks"),
            "CHUNK_STRATEGY": (chunking, "strategy"),
            "CHUNK_SIZE": (chunking, "chunk_size"),
            "CHUNK_OVERLAP": (chunking, "chunk_overlap"),
            "MIN_CHUNK_TOKENS": (chunking, "min_chunk_tokens"),
            "INCLUDE_HEADING_CONTEXT": (chunking, "include_heading_context"),
        }
        for suffix, (target, field) in mapping.items():
            value = env.get(ENV_PREFIX + suffix)
            if value is not None and value.strip() != "":
                target[field] = value.strip()
        try:
            return cls(
                ingestion=IngestionConfig.model_validate(ingestion),
                chunking=ChunkingConfig.model_validate(chunking),
            )
        except ValidationError as exc:
            raise ValueError(f"invalid RAG_* configuration: {exc}") from exc
