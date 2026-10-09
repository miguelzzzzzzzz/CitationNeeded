# TODO

Short-horizon task list. Milestones and acceptance criteria live in `PLAN.md`.

## M1 - Ingestion and chunking (done)
- [x] Packaging, tooling, CI (lint, format, strict typing, tests)
- [x] Document model and format-aware loaders (txt, md, html, pdf)
- [x] Configurable chunkers (fixed, recursive, structure-aware) with offsets and metadata
- [x] `rag-engine ingest` CLI with chunk statistics

## M2 - Indexing and retrieval primitives (in progress)
- [x] `Embedder` protocol and deterministic hashing embedder
- [x] fastembed adapter (`BAAI/bge-small-en-v1.5`) with `slow` tests and batching
- [x] `VectorStore` protocol and NumPy exact index (filters, upsert/delete by doc id, save/load)
- [x] BM25 index with hand-computed score tests
- [ ] `Retriever` returning scored chunks with provenance
- [ ] `rag-engine index` / `rag-engine search` CLI commands
