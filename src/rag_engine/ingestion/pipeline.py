"""Directory/file ingestion with per-file error isolation and de-duplication."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from rag_engine.config import IngestionConfig
from rag_engine.ingestion.loaders import DEFAULT_LOADERS, Loader, LoaderError, loader_for
from rag_engine.models import Document


@dataclass(frozen=True)
class SkippedFile:
    source: str
    reason: str


@dataclass
class IngestionReport:
    documents: list[Document] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)

    @property
    def files_seen(self) -> int:
        return len(self.documents) + len(self.skipped)


def _iter_files(root: Path, follow_symlinks: bool) -> Iterator[Path]:
    """Yield files under ``root`` in sorted order, skipping hidden entries."""
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink() and not follow_symlinks:
            continue
        if path.is_file():
            yield path


def ingest_path(
    root: str | Path,
    config: IngestionConfig | None = None,
    loaders: Iterable[Loader] = DEFAULT_LOADERS,
) -> IngestionReport:
    """Load every supported file under ``root``.

    Failures are isolated per file and reported in ``IngestionReport.skipped``
    (unsupported type, too large, unreadable, empty, or duplicate content).
    Sources are POSIX paths relative to ``root`` so ids are stable across
    machines.
    """
    config = config or IngestionConfig()
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"ingestion root does not exist: {root_path}")
    loader_list = tuple(loaders)
    base = root_path.parent if root_path.is_file() else root_path
    report = IngestionReport()
    seen_hashes: dict[str, str] = {}

    for path in _iter_files(root_path, config.follow_symlinks):
        source = path.relative_to(base).as_posix()
        loader = loader_for(path, loader_list)
        if loader is None:
            report.skipped.append(SkippedFile(source, f"unsupported extension {path.suffix!r}"))
            continue
        size = path.stat().st_size
        if size > config.max_file_bytes:
            report.skipped.append(
                SkippedFile(source, f"file too large ({size} > {config.max_file_bytes} bytes)")
            )
            continue
        try:
            document = loader.load(path, source)
        except LoaderError as exc:
            report.skipped.append(SkippedFile(source, str(exc)))
            continue
        duplicate_of = seen_hashes.get(document.content_hash)
        if duplicate_of is not None:
            report.skipped.append(SkippedFile(source, f"duplicate content of {duplicate_of}"))
            continue
        seen_hashes[document.content_hash] = source
        report.documents.append(document)
    return report
