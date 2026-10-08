"""Core data model shared by ingestion, chunking, retrieval, and answering.

Invariants (enforced by validators and tests):

* ``Section`` spans tile ``Document.text`` in order without overlapping.
* ``Chunk.text == document.text[chunk.start_char:chunk.end_char]``, which makes
  citations point at exact, verifiable source spans.
"""

from __future__ import annotations

import hashlib
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MetadataValue = str | int | float | bool | list[str] | None


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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

    ``doc_id`` is derived from the source path so re-ingesting a changed file
    updates the same logical document; ``content_hash`` detects the change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    source: str
    title: str
    format: str
    text: str
    content_hash: str
    sections: tuple[Section, ...]
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

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

    @classmethod
    def create(
        cls,
        *,
        source: str,
        title: str,
        format: str,
        text: str,
        sections: tuple[Section, ...] | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> Document:
        """Build a document, deriving ids and a default single section."""
        if sections is None:
            sections = (Section(start_char=0, end_char=len(text)),)
        return cls(
            doc_id=sha256_hex(source)[:16],
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

    @property
    def embedding_text(self) -> str:
        """Text to embed/index: heading context (if any) plus the chunk body."""
        if self.context_header:
            return f"{self.context_header}\n\n{self.text}"
        return self.text
