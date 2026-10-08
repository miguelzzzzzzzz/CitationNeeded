"""Document ingestion: loading, normalization, metadata extraction."""

from rag_engine.ingestion.loaders import (
    DEFAULT_LOADERS,
    HTMLLoader,
    Loader,
    LoaderError,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    loader_for,
)
from rag_engine.ingestion.normalize import normalize_text
from rag_engine.ingestion.pipeline import IngestionReport, SkippedFile, ingest_path

__all__ = [
    "DEFAULT_LOADERS",
    "HTMLLoader",
    "IngestionReport",
    "Loader",
    "LoaderError",
    "MarkdownLoader",
    "PDFLoader",
    "SkippedFile",
    "TextLoader",
    "ingest_path",
    "loader_for",
    "normalize_text",
]
