"""Format-aware document loaders.

Each loader turns one file into a normalized :class:`~rag_engine.models.Document`
with structural sections and extracted metadata. Loaders raise
:class:`LoaderError` for files they cannot handle; the ingestion pipeline
records those as skipped instead of aborting the whole run.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol

import yaml

from rag_engine.ingestion.normalize import normalize_text
from rag_engine.models import Document, MetadataValue, Section


class LoaderError(Exception):
    """Raised when a file cannot be turned into a document."""


class Loader(Protocol):
    format: str
    extensions: tuple[str, ...]

    def load(self, path: Path, source: str) -> Document: ...


# --------------------------------------------------------------------------- helpers


def read_text_file(path: Path) -> tuple[str, str]:
    """Read a text file, returning ``(text, encoding)``.

    Rejects binary content (NUL bytes) and falls back to latin-1 when the file
    is not valid UTF-8, so a single mis-encoded file never aborts ingestion.
    """
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        raise LoaderError("binary content (NUL bytes) in a text file")
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)


def split_front_matter(text: str) -> tuple[dict[str, MetadataValue], str]:
    """Split YAML front matter from Markdown. Invalid front matter is kept as text."""
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, text
    if not isinstance(data, dict):
        return {}, text
    metadata: dict[str, MetadataValue] = {}
    for key, value in data.items():
        if isinstance(value, str | int | float | bool) or value is None:
            metadata[str(key)] = value
        elif isinstance(value, list) and all(isinstance(v, str | int | float) for v in value):
            metadata[str(key)] = [str(v) for v in value]
        else:
            metadata[str(key)] = str(value)
    return metadata, text[match.end() :]


_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class Heading:
    offset: int
    level: int
    title: str


def find_markdown_headings(text: str) -> list[Heading]:
    """Find ATX headings, ignoring anything inside fenced code blocks."""
    headings: list[Heading] = []
    fence: str | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\n")
        fence_match = _FENCE.match(content)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0] * 3
            elif marker.startswith(fence):
                fence = None
        elif fence is None:
            heading = _ATX_HEADING.match(content)
            if heading:
                title = heading.group(2).strip()
                headings.append(Heading(offset, len(heading.group(1)), title))
        offset += len(line)
    return headings


def markdown_sections(text: str) -> tuple[tuple[Section, ...], list[Heading]]:
    """Split Markdown text into heading-delimited sections with heading paths."""
    headings = find_markdown_headings(text)
    if not headings:
        return (Section(start_char=0, end_char=len(text)),), headings
    sections: list[Section] = []
    if headings[0].offset > 0:
        sections.append(Section(start_char=0, end_char=headings[0].offset))
    stack: list[Heading] = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].offset if i + 1 < len(headings) else len(text)
        while stack and stack[-1].level >= heading.level:
            stack.pop()
        stack.append(heading)
        sections.append(
            Section(
                start_char=heading.offset,
                end_char=end,
                heading_path=tuple(h.title for h in stack),
            )
        )
    return tuple(sections), headings


def _base_metadata(path: Path, encoding: str | None = None) -> dict[str, MetadataValue]:
    metadata: dict[str, MetadataValue] = {
        "file_name": path.name,
        "extension": path.suffix.lower(),
        "file_size": path.stat().st_size,
    }
    if encoding is not None:
        metadata["encoding"] = encoding
    return metadata


def _require_text(text: str) -> str:
    if not text:
        raise LoaderError("no extractable text")
    return text


# --------------------------------------------------------------------------- loaders


class TextLoader:
    format = "text"
    extensions: tuple[str, ...] = (".txt", ".text", ".rst", ".log")

    def load(self, path: Path, source: str) -> Document:
        raw, encoding = read_text_file(path)
        text = _require_text(normalize_text(raw))
        metadata = _base_metadata(path, encoding)
        metadata["word_count"] = len(text.split())
        return Document.create(
            source=source, title=path.stem, format=self.format, text=text, metadata=metadata
        )


class MarkdownLoader:
    format = "markdown"
    extensions: tuple[str, ...] = (".md", ".markdown", ".mdx")

    def load(self, path: Path, source: str) -> Document:
        raw, encoding = read_text_file(path)
        front_matter, body = split_front_matter(raw.replace("\r\n", "\n"))
        text = _require_text(normalize_text(body))
        sections, headings = markdown_sections(text)
        metadata = _base_metadata(path, encoding)
        metadata.update(front_matter)
        metadata["word_count"] = len(text.split())
        metadata["heading_count"] = len(headings)
        title = _pick_title(front_matter.get("title"), headings, path)
        return Document.create(
            source=source,
            title=title,
            format=self.format,
            text=text,
            sections=sections,
            metadata=metadata,
        )


def _pick_title(explicit: MetadataValue, headings: Iterable[Heading], path: Path) -> str:
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    for heading in headings:
        if heading.level == 1:
            return heading.title
    return path.stem


class _HTMLToMarkdown(HTMLParser):
    """Minimal block-level HTML -> Markdown-like text converter.

    Headings become ATX headings so the Markdown sectioner can be reused.
    Known limitation: indentation inside ``<pre>`` blocks is not preserved.
    """

    SKIP = frozenset(
        {"script", "style", "noscript", "template", "svg", "nav", "footer", "aside", "form"}
    )
    BLOCK = frozenset(
        {
            "p", "div", "section", "article", "main", "header", "br", "tr", "table",
            "ul", "ol", "dl", "dt", "dd", "blockquote", "figure", "figcaption", "hr",
        }
    )  # fmt: skip
    HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self.meta: dict[str, MetadataValue] = {}
        self._skip_depth = 0
        self._in_title = False
        self._pre_depth = 0
        self._heading_level = 0
        self._heading_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {k: v or "" for k, v in attrs}
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if tag == "html" and attributes.get("lang"):
            self.meta["language"] = attributes["lang"]
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = (attributes.get("name") or attributes.get("property") or "").lower()
            if name in {"description", "author", "keywords", "og:title"} and attributes.get(
                "content"
            ):
                self.meta[name.replace("og:", "og_")] = attributes["content"].strip()
        elif tag in self.HEADINGS and self._skip_depth == 0:
            self._heading_level = int(tag[1])
            self._heading_text = []
        elif tag == "pre":
            self._pre_depth += 1
            self.parts.append("\n\n```\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        elif tag in self.HEADINGS and self._heading_level:
            heading = " ".join("".join(self._heading_text).split())
            if heading:
                self.parts.append(f"\n\n{'#' * self._heading_level} {heading}\n\n")
            self._heading_level = 0
        elif tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
            self.parts.append("\n```\n\n")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        if self._heading_level:
            self._heading_text.append(data)
            return
        if self._pre_depth:
            self.parts.append(data)
            return
        collapsed = " ".join(data.split())
        if collapsed:
            # keep a separating space where the source had whitespace
            lead = " " if data[:1].isspace() else ""
            trail = " " if data[-1:].isspace() else ""
            self.parts.append(f"{lead}{collapsed}{trail}")


class HTMLLoader:
    format = "html"
    extensions: tuple[str, ...] = (".html", ".htm", ".xhtml")

    def load(self, path: Path, source: str) -> Document:
        raw, encoding = read_text_file(path)
        parser = _HTMLToMarkdown()
        parser.feed(raw)
        parser.close()
        text = "\n".join(line.strip() for line in "".join(parser.parts).split("\n"))
        text = _require_text(normalize_text(text))
        sections, headings = markdown_sections(text)
        metadata = _base_metadata(path, encoding)
        metadata.update(parser.meta)
        metadata["word_count"] = len(text.split())
        metadata["heading_count"] = len(headings)
        title = " ".join(parser.title.split()) or _pick_title(None, headings, path)
        return Document.create(
            source=source,
            title=title,
            format=self.format,
            text=text,
            sections=sections,
            metadata=metadata,
        )


class PDFLoader:
    """Extracts text per page; each page becomes a section with ``page`` set."""

    format = "pdf"
    extensions: tuple[str, ...] = (".pdf",)

    def load(self, path: Path, source: str) -> Document:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted and not reader.decrypt(""):
                raise LoaderError("encrypted PDF")
            pages = [page.extract_text() or "" for page in reader.pages]
        except (PdfReadError, ValueError, KeyError, OSError) as exc:
            raise LoaderError(f"unreadable PDF: {exc}") from exc

        parts: list[str] = []
        sections: list[Section] = []
        cursor = 0
        empty_pages = 0
        for number, page_text in enumerate(pages, start=1):
            cleaned = normalize_text(page_text, dehyphenate=True)
            if not cleaned:
                empty_pages += 1
                continue
            if parts:
                cursor += 2  # "\n\n" page separator
            sections.append(Section(start_char=cursor, end_char=cursor + len(cleaned), page=number))
            parts.append(cleaned)
            cursor += len(cleaned)
        if not parts:
            raise LoaderError("no extractable text (scanned PDF? OCR is not supported)")
        text = "\n\n".join(parts)

        info = reader.metadata
        pdf_title = str(info.title).strip() if info is not None and info.title else ""
        metadata = _base_metadata(path)
        metadata["page_count"] = len(pages)
        metadata["empty_pages"] = empty_pages
        metadata["word_count"] = len(text.split())
        if info is not None and info.author:
            metadata["author"] = str(info.author)
        return Document.create(
            source=source,
            title=pdf_title or path.stem,
            format=self.format,
            text=text,
            sections=tuple(sections),
            metadata=metadata,
        )


DEFAULT_LOADERS: tuple[Loader, ...] = (TextLoader(), MarkdownLoader(), HTMLLoader(), PDFLoader())


def loader_for(path: Path, loaders: Iterable[Loader] = DEFAULT_LOADERS) -> Loader | None:
    suffix = path.suffix.lower()
    for loader in loaders:
        if suffix in loader.extensions:
            return loader
    return None
