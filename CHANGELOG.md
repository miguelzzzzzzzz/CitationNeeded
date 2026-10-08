# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Renamed the repository to `CitationNeeded` (display title "Citation Needed"). The distribution, import package, and CLI names are unchanged.

### Added
- Retrieval primitives (M2, part 1): `Embedder` protocol, deterministic `HashingEmbedder` (signed feature hashing, offline test double and lexical baseline), and `InMemoryVectorStore` with exact cosine search, pre-ranking metadata filters, deterministic tie-breaking, document-level upsert/delete, and validated save/load.
- `rag-engine ingest` CLI writing chunks as JSONL with statistics and skipped-file reasons (overrides are re-validated; invalid options exit with code 2); `scripts/compare_chunking.py`; `docs/ingestion-and-chunking.md`.
- Chunking strategies: fixed token windows, recursive separator splitting (paragraph > line > sentence > word), and structure-aware chunking within sections; small-chunk merging, heading-path and page provenance, optional heading context for embedding text, deterministic content-sensitive chunk ids, and chunk statistics.
- Document model (`Document`, `Section`, `Chunk`) with validated, non-overlapping section spans, stable path-derived ids, and content hashes.
- Loaders for plain text, Markdown (YAML front matter, ATX headings outside code fences), HTML (title/meta extraction, script/nav/footer removal, headings to sections), and PDF (per-page sections, de-hyphenation, empty-page accounting).
- `ingest_path`: deterministic directory walk with per-file error isolation (unsupported, oversized, binary, empty, unreadable, duplicate content).
- Project specification, milestone plan, and ADRs 0001-0005 (vector index, local models, evaluation corpus, LLM provider interface, token budgets).
- Project scaffold: `pyproject.toml` (hatchling), MIT license, `.env.example`, `.gitignore`.
- Validated runtime settings (`rag_engine.config`) loaded from `RAG_*` environment variables.
- GitHub Actions CI: ruff lint + format check, strict mypy, pytest with coverage on Python 3.11 and 3.13.
