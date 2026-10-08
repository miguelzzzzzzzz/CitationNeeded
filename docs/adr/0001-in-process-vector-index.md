# ADR-0001: In-process NumPy exact vector index behind a `VectorStore` protocol

Status: Accepted (2026-10-09)

## Context
Target scale is up to ~100k chunks; the evaluation corpus (SciFact) is ~5k
documents. Results must be exactly reproducible so benchmark deltas reflect
retrieval changes, not ANN approximation noise. Docker is not available on the
development machine, which makes running a database service for every test or
eval inconvenient.

## Decision
Implement a `VectorStore` protocol and ship an in-process exact index:
L2-normalized float32 matrix, cosine similarity via a single matrix-vector
product, metadata filtering, and persistence to `.npy` + JSONL. Brute-force
search over 100k x 384 float32 vectors is ~150 MB and a single BLAS call,
which is adequate at this scale.

## Alternatives considered
- **Qdrant (local mode or server)**: good filtering and HNSW, but adds a
  dependency and approximate results; a strong candidate for a second adapter.
- **pgvector**: right choice if the project needed PostgreSQL for metadata
  anyway; it does not today.
- **FAISS**: fast, but heavy native wheels and little benefit at this scale.

## Consequences
Zero infrastructure and deterministic evals. Memory grows linearly with corpus
size and there is no concurrent-writer story; a server-backed adapter
(Qdrant or pgvector) can be added behind the same protocol if scale demands it.
