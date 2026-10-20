# ADR-0007: A read-only HTTP service over a prebuilt index

- Status: Accepted
- Date: 2026-10-20
- Context: milestone M6 (FastAPI service, Docker image)

## Context

The original M6 plan listed `POST /ingest`, `POST /search`, `POST /answer` and
`GET /health`. Ingesting over HTTP means the service writes its own index while
it serves queries: it needs a write lock, an atomic directory swap so a crash
never leaves a half-written index, a policy for concurrent readers during the
swap, upload size limits, and a way to accept binary formats (PDF) over JSON.
Indexing is also the slow, CPU-heavy part (embedding every chunk), which is a
poor fit for a request/response call.

## Decision

1. The service is read-only. It loads one index directory (`RAG_INDEX_DIR`,
   written by `rag-engine index`) once at startup and serves `GET /health`,
   `POST /search` and `POST /answer`. Rebuilding the index means re-running
   `rag-engine index` and restarting the service. `POST /ingest` is not part
   of M6.
2. `GET /health` is both liveness and readiness: 200 while an index is served,
   503 (with the reason) when none is configured or the directory has no
   `index.json` yet. A directory that has a manifest but fails to load stops
   startup, so a corrupt index fails loudly instead of being reported as "not
   built yet".
3. The request schema enforces the same option rules as the CLI (for example,
   fusion options only in hybrid mode). Errors map to 422 (schema), 400 (a
   schema-valid request the server cannot serve, such as the `openai`
   generator without `RAG_LLM_*`), 503 (no index, or an optional dependency
   such as fastembed is missing), and 502 (the answer generator failed). A
   `ValueError` raised during retrieval is a 500: it points at the index or
   the code, not at the request.
4. `rerank: true` uses the one cross-encoder configured on the server
   (`RAG_RERANK_MODEL`); clients cannot name a model, so a request cannot make
   the server download arbitrary weights.
5. The Docker image runs as a non-root user (uid 10001) and is built and
   smoke-tested in CI with the offline hashing embedder (no model downloads):
   an index is built inside the container, served from a read-only mount, and
   queried through every endpoint.

## Consequences

- Concurrent requests only read shared state; no locking is needed around the
  index. The reranker and a fastembed query embedder load their models lazily
  on first use, and that first load is not synchronized: concurrent first
  requests may each load the model once (results are unaffected).
- Scaling out is a matter of running more read-only replicas against the same
  index volume.
- An HTTP ingest endpoint can be added later as a separate, write-side service
  without changing the read path.
