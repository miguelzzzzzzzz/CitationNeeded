"""BEIR SciFact loader used by the milestone M4 retrieval evaluation.

Implements ADR-0003: evaluation corpora are fetched on demand into an untracked
cache directory instead of being vendored into the repository. SciFact claims
are licensed CC BY 4.0 and the paper abstracts ODC-By 1.0, so the dataset must
never be committed to this repository.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = [
    "SCIFACT_MD5",
    "SCIFACT_NAME",
    "SCIFACT_URL",
    "SciFactDocument",
    "SciFactQuery",
    "SciFactSplit",
    "download_scifact",
    "ensure_scifact",
    "load_scifact",
    "md5_file",
]

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
SCIFACT_NAME = "scifact"

#: BEIR archives ship a single top-level directory named after the dataset.
_ZIP_NAME = f"{SCIFACT_NAME}.zip"
_SPLITS = ("test", "train")


class SciFactDocument(BaseModel):
    """One SciFact corpus entry: a paper title plus its abstract."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    text: str


class SciFactQuery(BaseModel):
    """One SciFact claim, to be retrieved against the corpus."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    text: str


class SciFactSplit(BaseModel):
    """One SciFact split loaded in memory, ready for scoring."""

    model_config = ConfigDict(frozen=True)

    name: str
    corpus: dict[str, SciFactDocument]
    queries: dict[str, SciFactQuery]
    qrels: dict[str, dict[str, int]]
    checksum: str
    root: Path


def md5_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex MD5 digest of ``path``, read in ``chunk_size`` blocks."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_scifact(
    dest_dir: Path,
    *,
    url: str = SCIFACT_URL,
    expected_md5: str = SCIFACT_MD5,
    force: bool = False,
) -> Path:
    """Download and unpack SciFact, returning the extracted ``scifact`` directory.

    ``dest_dir`` is the shared dataset cache (e.g. ``evals/datasets``); the archive
    lives at ``dest_dir/scifact.zip`` and the payload at ``dest_dir/scifact``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / _ZIP_NAME
    extracted = dest_dir / SCIFACT_NAME

    if zip_path.is_file():
        cached_md5 = md5_file(zip_path)
        if cached_md5 == expected_md5 and not force and _looks_extracted(extracted):
            return extracted
        if force or cached_md5 != expected_md5:
            zip_path.unlink()

    if not zip_path.is_file():
        _download_zip(url, zip_path, expected_md5)

    if force or not _looks_extracted(extracted):
        if extracted.is_dir():
            shutil.rmtree(extracted)
        with zipfile.ZipFile(zip_path) as archive:
            _safe_extract(archive, dest_dir)
        if not extracted.is_dir():
            raise ValueError(f"archive did not contain {SCIFACT_NAME}/ after extraction")

    return extracted


def load_scifact(
    root: Path,
    *,
    split: str = "test",
    checksum: str | None = None,
) -> SciFactSplit:
    """Read an extracted SciFact directory into memory.

    Only queries listed in ``qrels/{split}.tsv`` are kept, per the BEIR scoring
    convention; the corpus is retained in full so every document can be ranked.
    """
    if split not in _SPLITS:
        raise ValueError(f"unknown SciFact split {split!r}; expected 'test' or 'train'")

    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels" / f"{split}.tsv"
    for path in (corpus_path, queries_path, qrels_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"SciFact {split!r} split is incomplete: missing {path}. "
                f"Run download_scifact({root.parent!s}) first."
            )

    corpus = _load_corpus(corpus_path)
    queries = {query.query_id: query for query in _load_queries(queries_path)}
    qrels = _load_qrels(qrels_path)
    # qrels define the evaluation set: drop queries with no judgments at all.
    kept = {query_id: query for query_id, query in queries.items() if query_id in qrels}

    return SciFactSplit(
        name=split,
        corpus=corpus,
        queries=kept,
        qrels=qrels,
        checksum=checksum if checksum is not None else _sibling_checksum(root),
        root=root,
    )


def ensure_scifact(
    dest_dir: Path,
    *,
    split: str = "test",
    force: bool = False,
    url: str = SCIFACT_URL,
    expected_md5: str = SCIFACT_MD5,
) -> SciFactSplit:
    """Download SciFact if needed and return the requested split in memory."""
    root = download_scifact(dest_dir, url=url, expected_md5=expected_md5, force=force)
    checksum = md5_file(dest_dir / _ZIP_NAME)
    return load_scifact(root, split=split, checksum=checksum)


def _safe_extract(archive: zipfile.ZipFile, dest_dir: Path) -> None:
    """Extract ``archive`` into ``dest_dir``, rejecting absolute or ``..`` members."""
    dest = dest_dir.resolve()
    for name in archive.namelist():
        posix = PurePosixPath(name)
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"unsafe zip entry: {name}")
        target = (dest_dir / name).resolve()
        if target != dest and dest not in target.parents:
            raise ValueError(f"unsafe zip entry: {name}")
    archive.extractall(dest_dir)


def _looks_extracted(root: Path) -> bool:
    """An extracted directory counts as complete once its corpus file is present."""
    return (root / "corpus.jsonl").is_file()


def _download_zip(url: str, zip_path: Path, expected_md5: str) -> None:
    """Stream ``url`` into ``zip_path`` via a ``.partial`` file, verifying the digest."""
    partial = zip_path.with_name(f"{zip_path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        actual_md5 = md5_file(partial)
        if actual_md5 != expected_md5:
            raise ValueError(
                f"checksum mismatch for {url}: expected {expected_md5}, got {actual_md5}"
            )
        partial.replace(zip_path)
    finally:
        # Covers both the failure path and the (already-renamed) success path.
        partial.unlink(missing_ok=True)


def _sibling_checksum(root: Path) -> str:
    """Best-effort md5 of the archive that produced ``root`` (``""`` when absent)."""
    archive = root.parent / _ZIP_NAME
    return md5_file(archive) if archive.is_file() else ""


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one decoded JSON object per non-blank line of ``path``."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"expected one JSON object per line in {path}")
            yield record


def _as_text(value: object) -> str:
    """Coerce an optional JSON string field to ``str`` (missing becomes ``""``)."""
    return value if isinstance(value, str) else ""


def _as_id(value: object) -> str:
    """Coerce a JSON identifier to ``str``, ignoring absent or structured values."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _is_int(value: str) -> bool:
    """Return whether ``value`` parses as a base-10 integer."""
    try:
        int(value)
    except ValueError:
        return False
    return True


def _load_corpus(path: Path) -> dict[str, SciFactDocument]:
    corpus: dict[str, SciFactDocument] = {}
    for record in _read_jsonl(path):
        doc_id = _as_id(record.get("_id") or record.get("id"))
        if doc_id:
            corpus[doc_id] = SciFactDocument(
                doc_id=doc_id,
                title=_as_text(record.get("title")),
                text=_as_text(record.get("text")),
            )
    return corpus


def _load_queries(path: Path) -> list[SciFactQuery]:
    queries: list[SciFactQuery] = []
    for record in _read_jsonl(path):
        query_id = _as_id(record.get("_id") or record.get("id"))
        if query_id:
            queries.append(SciFactQuery(query_id=query_id, text=_as_text(record.get("text"))))
    return queries


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.reader(handle, delimiter="\t")):
            if len(row) < 3:
                continue
            query_id, doc_id, raw_grade = row[0].strip(), row[1].strip(), row[2].strip()
            if index == 0 and not _is_int(raw_grade):
                continue  # Header row: "query-id<TAB>corpus-id<TAB>score".
            if not query_id or not doc_id:
                continue
            qrels.setdefault(query_id, {})[doc_id] = int(raw_grade)
    return qrels
