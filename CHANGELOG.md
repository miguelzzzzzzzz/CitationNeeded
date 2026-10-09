# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Renamed the repository to `CitationNeeded` (display title "Citation Needed"). The distribution, import package, and CLI names are unchanged.

### Added
- `rag-engine index` (documents or an `ingest` chunks JSONL to an index directory; `--embedder` spec) and `rag-engine search` (dense or lexical mode, `-k`, repeatable `KEY=VALUE` filters, `--json`), with end-to-end tests.
- Retrievers (M2): `Retriever` protocol, `DenseRetriever` (embedder + exact vector index) and `LexicalRetriever` (BM25), both returning `RetrievedChunk` results with rank, score, retriever name, and provenance (doc/chunk id, source, heading path, page, character span). Index directories (`save_index`/`load_index`) hold both indexes plus a manifest naming the embedder, which `embedder_from_spec` rebuilds on load. Hybrid fusion is deferred to M3.
- `BM25Index` (M2): Okapi BM25 (k1 = 1.2, b = 0.75 by default; non-negative Lucene IDF) over an in-memory inverted index of each chunk's heading context plus body; NFKC + casefold word tokenizer with optional stopwords; document-level upsert/delete that keep document frequencies and average length exact; metadata filters before top-k; insertion-order tie-breaking; JSON save/load that rebuilds postings and detects tokenizer mismatch. Tests check scores against values worked out by hand.
- `FastEmbedEmbedder` (M2): `BAAI/bge-small-en-v1.5` through fastembed (optional `embeddings` extra, lazy import), bounded batches written into a preallocated float32 matrix, zero rows for empty texts without model calls, explicit L2 normalization, optional query prefix, and checks for vector count, dimension, and NaN/inf output. `EmbeddingConfig` with `RAG_EMBEDDING_*` variables. Real-model tests are marked `slow`.
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

### Fixed
- Index persistence (M2 review): `save_index` refuses embedders whose name is not a reloadable spec and retrievers whose chunk-id sets differ, and removes an old manifest before rewriting; `load_index` validates manifest fields, checks that both halves hold the same chunk ids, accepts the build-time BM25 tokenizer, and passes the manifest dimension to the embedder so lexical search on a fastembed-built index no longer needs fastembed.
- CLI (M2 review): a missing fastembed install is reported as `error: ...` (exit 2) instead of a traceback; chunking options passed with a chunks `.jsonl` input produce a warning since they have no effect.
- Chunking overrides on the command line no longer reset `RAG_EMBEDDING_*` (and other non-chunking) settings to defaults.
- Dense retrieval returns no hits for a query whose embedding is all zeros instead of an arbitrary list of zero-score chunks.
