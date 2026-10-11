"""Structured answer schema for M5 answer generation with citations.

Defines the pydantic v2 response models shared by every ``AnswerGenerator``
implementation (extractive default and the OpenAI-compatible client) as
specified by ADR-0004. Keeping the schema free of retrieval imports avoids a
module cycle between ``rag_engine.answering`` and ``rag_engine.retrieval``; the
``citation_from_provenance`` factory takes plain keyword arguments instead.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ANSWER_SCHEMA_VERSION: int = 1

__all__ = ["ANSWER_SCHEMA_VERSION", "AnswerResponse", "Citation", "citation_from_provenance"]


class Citation(BaseModel):
    """Verifiable span citation pointing back at a single retrieved chunk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    chunk_id: str
    source: str | None = None
    content_hash: str | None = None
    heading_path: tuple[str, ...] = ()
    page: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    quote: str

    @field_validator("quote")
    @classmethod
    def _validate_quote(cls, value: str) -> str:
        # Whitespace-only quotes are not verifiable evidence and would produce
        # citations that cannot be checked against the source text.
        if not value.strip():
            raise ValueError("quote must contain non-whitespace text")
        return value

    @model_validator(mode="after")
    def _validate_span(self) -> Citation:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


class AnswerResponse(BaseModel):
    """Answer plus its supporting citations, returned by every generator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = ANSWER_SCHEMA_VERSION
    query: str
    answer: str
    citations: tuple[Citation, ...] = ()
    generator: str
    abstained: bool = False
    model_name: str | None = None

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value

    @model_validator(mode="after")
    def _validate_consistency(self) -> AnswerResponse:
        if self.schema_version != ANSWER_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {ANSWER_SCHEMA_VERSION}, got {self.schema_version}"
            )
        if self.abstained and self.citations:
            raise ValueError("abstained answers must not carry citations")
        if not self.abstained and not self.answer.strip():
            raise ValueError("a non-abstained answer must contain non-whitespace text")
        return self


def citation_from_provenance(
    *,
    doc_id: str,
    chunk_id: str,
    quote: str,
    source: str | None = None,
    content_hash: str | None = None,
    heading_path: tuple[str, ...] | list[str] = (),
    page: int | None = None,
    page_end: int | None = None,
    start_char: int,
    end_char: int,
) -> Citation:
    """Build a :class:`Citation` from provenance fields plus the quoted span."""
    return Citation(
        doc_id=doc_id,
        chunk_id=chunk_id,
        source=source,
        content_hash=content_hash,
        heading_path=tuple(heading_path),
        page=page,
        page_end=page_end,
        start_char=start_char,
        end_char=end_char,
        quote=quote,
    )
