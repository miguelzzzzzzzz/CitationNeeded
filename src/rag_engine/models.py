"""Core data model shared by ingestion, chunking, retrieval, and answering.

Invariants (enforced by validators and tests):

* ``Document.doc_id == make_doc_id(corpus_id, source)`` and
  ``Document.content_hash == sha256_hex(text)``, so a document reloaded from
  disk can be verified instead of trusted.
* ``Section`` spans tile ``Document.text`` in order without overlapping.
* ``len(Chunk.text) == Chunk.end_char - Chunk.start_char`` and
  ``Chunk.text == document.text[chunk.start_char:chunk.end_char]``, which makes
  citations point at exact, verifiable source spans.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MetadataValue = str | int | float | bool | list[str] | None

DEFAULT_CORPUS_ID = "default"

_CORPUS_ID_MAX_LENGTH = 64
_CORPUS_ID_RE = re.compile(rf"^[a-z0-9][a-z0-9._-]{{0,{_CORPUS_ID_MAX_LENGTH - 1}}}$")
_INVALID_CORPUS_ID_CHARS_RE = re.compile(r"[^a-z0-9._-]+")
_LEADING_NON_ALNUM_RE = re.compile(r"^[^a-z0-9]+")


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_corpus_id(value: str) -> str:
    """Return ``value`` if it is a valid corpus id, else raise ``ValueError``.

    ``:`` can never appear in a valid id, which keeps the ``"<corpus_id>:<source>"``
    doc-id scheme unambiguous.
    """
    if _CORPUS_ID_RE.fullmatch(value) is None:
        raise ValueError(f"corpus_id must match {_CORPUS_ID_RE.pattern!r}, got {value!r}")
    return value


def corpus_id_from_name(name: str) -> str:
    """Slugify a directory name into a valid corpus id (never empty)."""
    slug = _INVALID_CORPUS_ID_CHARS_RE.sub("-", name.lower())
    slug = _LEADING_NON_ALNUM_RE.sub("", slug).rstrip("-")[:_CORPUS_ID_MAX_LENGTH]
    return slug or DEFAULT_CORPUS_ID


def make_doc_id(corpus_id: str, source: str) -> str:
    """Stable document id: first 16 hex chars of sha256("<corpus_id>:<source>").

    ``source`` is the POSIX path relative to the corpus root; the corpus id keeps
    identical relative paths from different corpora from colliding.
    """
    return sha256_hex(f"{validate_corpus_id(corpus_id)}:{validate_source(source)}")[:16]


def validate_source(source: str) -> str:
    """Require a normalized POSIX relative path, so one file has exactly one id.

    ``./a.md``, ``a\\b.md``, ``/a.md`` and ``a//b.md`` are rejected rather than
    hashed to a second id for the same file.
    """
    parts = source.split("/")
    if (
        not source
        or "\\" in source
        or source.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"source must be a normalized relative POSIX path, got {source!r}")
    return source


class Section(BaseModel):
    """A contiguous span of a document with structural context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    heading_path: tuple[str, ...] = ()
    page: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.end_char < self.start_char:
            raise ValueError("section end_char must be >= start_char")
        return self


class Document(BaseModel):
    """A normalized source document.

    ``doc_id`` is the first 16 hex chars of ``sha256("<corpus_id>:<source>")``
    where ``source`` is the POSIX path relative to the corpus root: the id is
    stable across machines and ingest roots as long as ``corpus_id`` and that
    relative path are the same, so re-ingesting a changed file updates the same
    logical document while ``content_hash`` detects the change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    corpus_id: str = DEFAULT_CORPUS_ID
    source: str
    title: str
    format: str
    text: str
    content_hash: str
    sections: tuple[Section, ...]
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @field_validator("corpus_id")
    @classmethod
    def _check_corpus_id(cls, value: str) -> str:
        return validate_corpus_id(value)

    @model_validator(mode="after")
    def _check_sections(self) -> Self:
        cursor = 0
        for section in self.sections:
            if section.start_char < cursor:
                raise ValueError("sections must be ordered and non-overlapping")
            if section.end_char > len(self.text):
                raise ValueError("section extends past end of document text")
            cursor = section.end_char
        return self

    @model_validator(mode="after")
    def _check_derived_fields(self) -> Self:
        expected_doc_id = make_doc_id(self.corpus_id, self.source)
        if self.doc_id != expected_doc_id:
            raise ValueError(
                f"doc_id {self.doc_id!r} does not match "
                f"make_doc_id(corpus_id, source) = {expected_doc_id!r}"
            )
        expected_hash = sha256_hex(self.text)
        if self.content_hash != expected_hash:
            raise ValueError(
                f"content_hash {self.content_hash!r} does not match "
                f"sha256_hex(text) = {expected_hash!r}"
            )
        return self

    @classmethod
    def create(
        cls,
        *,
        source: str,
        title: str,
        format: str,
        text: str,
        corpus_id: str = DEFAULT_CORPUS_ID,
        sections: tuple[Section, ...] | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> Document:
        """Build a document, deriving ids and a default single section."""
        if sections is None:
            sections = (Section(start_char=0, end_char=len(text)),)
        return cls(
            doc_id=make_doc_id(corpus_id, source),
            corpus_id=corpus_id,
            source=source,
            title=title,
            format=format,
            text=text,
            content_hash=sha256_hex(text),
            sections=sections,
            metadata=dict(metadata or {}),
        )

    def section_at(self, char_offset: int) -> Section | None:
        """Return the section containing ``char_offset`` (None if outside all)."""
        for section in self.sections:
            if section.start_char <= char_offset < section.end_char:
                return section
        return None


class Chunk(BaseModel):
    """A retrievable unit of text with full provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    doc_id: str
    index: int = Field(ge=0)
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    token_count: int = Field(ge=0)
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    context_header: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_offsets(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError(
                f"chunk end_char {self.end_char} must be > start_char {self.start_char}"
            )
        span = self.end_char - self.start_char
        if len(self.text) != span:
            raise ValueError(
                f"chunk text length {len(self.text)} must equal end_char - start_char = {span}"
            )
        return self

    @property
    def embedding_text(self) -> str:
        """Text to embed/index: heading context (if any) plus the chunk body."""
        if self.context_header:
            return f"{self.context_header}\n\n{self.text}"
        return self.text
