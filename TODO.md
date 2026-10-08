# TODO

Short-horizon task list. Milestones and acceptance criteria live in `PLAN.md`.

## M1 - Ingestion and chunking (in progress)
- [x] Packaging, tooling, CI (lint, format, strict typing, tests)
- [x] Document model and format-aware loaders (txt, md, html, pdf)
- [x] Configurable chunkers (fixed, recursive, structure-aware) with offsets and metadata
- [x] `rag-engine ingest` CLI with chunk statistics

## M2 - Indexing and retrieval primitives (next)
- [ ] `Embedder` protocol, deterministic hashing embedder for tests, fastembed adapter (`slow` tests)
- [ ] `VectorStore` protocol and NumPy exact index (filters, upsert/delete by doc id, save/load)
- [ ] BM25 index with hand-computed score tests
- [ ] `Retriever` returning scored chunks with provenance
