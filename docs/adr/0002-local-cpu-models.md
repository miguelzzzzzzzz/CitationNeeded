# ADR-0002: Local CPU embedding and reranking models (ONNX via fastembed)

Status: Accepted (2026-10-09)

## Context
No paid model API is authorized. CI runners and the development box are
CPU-only with limited memory, and PyTorch would add a multi-GB dependency.

## Decision
Use fastembed (ONNX Runtime) with small open models:
`BAAI/bge-small-en-v1.5` (384-dim, MIT) for embeddings and a MiniLM MS MARCO
cross-encoder for reranking. Models sit behind `Embedder`/`Reranker`
protocols; tests use a deterministic hashing embedder so the default test
suite needs no downloads. Tests that need real models are marked `slow`.

## Alternatives considered
- **sentence-transformers**: richer model zoo but pulls in PyTorch.
- **Hosted embedding APIs**: better quality ceiling but costs money and makes
  evals non-reproducible offline.

## Consequences
Zero marginal cost and offline reproducibility. Quality is below large hosted
models; the eval harness makes the gap measurable if a provider is added later.
