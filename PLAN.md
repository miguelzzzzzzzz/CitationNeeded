# PLAN

Milestones are ordered; each ends with green CI and an updated CHANGELOG.

| ID | Milestone | Status |
| --- | --- | --- |
| M1 | Project infrastructure, ingestion, configurable chunking | done |
| M2 | Embeddings, vector store, BM25 index, retrieval interface | done |
| M3 | Hybrid fusion (RRF / weighted), metadata filters, cross-encoder reranking | done |
| M4 | Evaluation harness (SciFact + structured-doc set), benchmark reports | done |
| M5 | Answer generation: provider interface, extractive fallback, citations, structured output | planned |
| M6 | FastAPI service, Dockerfile, docker build in CI | planned |
| M7 | Hardening: error handling, perf (batching, caching), docs, full release (v0.1.0 was the M1-M3 interface release) | planned |

## M1 - Infrastructure, ingestion, chunking

Scope
- Packaging (src layout), ruff, strict mypy, pytest, GitHub Actions CI.
- Document model: `Document` with normalized text, `Section`s (heading path,
  page, char span), metadata, content hash, stable id.
- Loaders: plain text, Markdown (front matter, ATX headings outside code
  fences), HTML (title/meta, headings, skip script/style/nav), PDF (per page).
- Chunkers: fixed token window, recursive separator splitting, structure-aware
  (per section); configurable size/overlap/min size; heading context.
- CLI: `rag-engine ingest PATH --out chunks.jsonl` with statistics.

Acceptance
- Offset invariant holds for every chunk of every strategy (tested).
- No chunk exceeds `chunk_size` tokens (tested), overlap respected.
- Unsupported, empty, oversized, and undecodable files are reported, not fatal.
- CI green.

## M2 - Indexing and retrieval primitives

- `Embedder` protocol; fastembed adapter; deterministic hashing embedder for tests.
- `VectorStore` protocol; NumPy exact cosine index with metadata filters and
  save/load; upsert/delete by document id.
- BM25 lexical index (in-repo implementation, tested against hand-computed scores).
- `Retriever` returning scored chunks with provenance.

## M3 - Hybrid retrieval and reranking

- RRF and weighted score fusion; metadata filter pass-through.
- `Reranker` protocol with local cross-encoder; top-N rerank budget.

## M4 - Evaluation

- Dataset loader for BEIR SciFact (download + checksum), structured-doc eval set.
- Metric functions (Recall@K, Hit@K, MRR, nDCG) with unit tests.
- Runner producing JSON reports; comparison table generator for README.
- Status (2026-10-10): harness landed (`rag_engine.eval`, `rag-engine evaluate`,
  `scripts/eval_table.py`). Structured-doc smoke report committed under
  `evals/results/` (hashing embedder — not a retrieval-quality claim). Full
  SciFact + fastembed benchmark optional follow-up; do not put numbers in the
  README until a committed SciFact report exists.

## M5 - Answers

- `AnswerGenerator` protocol; OpenAI-compatible client (env-configured);
  extractive generator (sentence selection with citations) as default.
- Pydantic response schema; malformed-output handling and repair.

## M6 - Service and deployment

- FastAPI endpoints: `POST /ingest`, `POST /search`, `POST /answer`, `GET /health`.
- Dockerfile (multi-stage, non-root), compose file, CI job building the image.

## M7 - Hardening and release

- Batching and caching of embeddings, latency profiling, structured logging.
- Final README with benchmark tables from reports; tag the full release (v0.1.0, the M1-M3 interface release, was cut on 2026-10-09).
