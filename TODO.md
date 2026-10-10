# TODO

Short-horizon task list. Milestones and acceptance criteria live in `PLAN.md`.

## M1 - Ingestion and chunking (done)
- [x] Packaging, tooling, CI (lint, format, strict typing, tests)
- [x] Document model and format-aware loaders (txt, md, html, pdf)
- [x] Configurable chunkers (fixed, recursive, structure-aware) with offsets and metadata
- [x] `rag-engine ingest` CLI with chunk statistics

## M2 - Indexing and retrieval primitives (done)
- [x] `Embedder` protocol and deterministic hashing embedder
- [x] fastembed adapter (`BAAI/bge-small-en-v1.5`) with `slow` tests and batching
- [x] `VectorStore` protocol and NumPy exact index (filters, upsert/delete by doc id, save/load)
- [x] BM25 index with hand-computed score tests
- [x] `Retriever` returning scored chunks with provenance
- [x] `rag-engine index` / `rag-engine search` CLI commands

## M3 - Hybrid retrieval and reranking (done)
- [x] Reciprocal rank fusion and weighted (per-list min-max) score fusion, tests with hand-computed scores
- [x] `HybridRetriever` passing the same metadata filters to every retriever before ranking
- [x] `Reranker` protocol, fastembed cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`), `RerankingRetriever` with a candidate budget
- [x] `rag-engine search --mode hybrid --fusion/--weight/--rrf-k/--candidates --rerank`
- [x] Lexical search on a fastembed-built index works without fastembed installed


## M4 - Evaluation harness (done)
- [x] SciFact loader (download + MD5 `5f7d1de60b170fc8027bb7898e2efca1`), data under `evals/datasets/` (gitignored)
- [x] Structured-doc eval set under `evals/structured/` with hand-labeled qrels
- [x] Metrics: Recall@{1,5,10}, Hit@{1,5,10}, MRR@10, nDCG@10 with hand-computed unit tests
- [x] Runner + `rag-engine evaluate` writing JSON reports (config, checksum, git SHA, hardware)
- [x] `scripts/eval_table.py` comparison table generator
- [ ] Optional: full SciFact run with fastembed (±rerank) and README table copied from that report

## Review of 84bce05 (done, see REVIEW.md)
- [x] Corpus-scoped document ids (`corpus_id` + relative path) with pinned-id, stability and no-collision tests
- [x] `documents.jsonl` beside chunk files and in the index; citations verified on build and load; index format 2
- [x] `content_hash` and `page_end` in provenance; chunk offset validation; NaN/inf weights rejected
- [x] Per-stage `components`/`ranks` (nested rerank keeps every stage); `schema_version` on `search --json`
- [x] End-to-end offset-invariant tests across loaders and pipeline stages
