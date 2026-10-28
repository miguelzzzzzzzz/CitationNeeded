# PROJECT_SPEC: Citation Needed (production-rag-engine)

Status: active, milestone M1 (ingestion and chunking). Last revised: 2026-10-09.

## 1. Problem statement

Most RAG examples are "chat with a PDF": one loader, fixed-size chunks, one
embedding model, top-k cosine search, and a prompt. They give no evidence that
retrieval works, no way to compare design choices, and no path to running as a
service. This project builds a RAG engine the way a production team would:
every stage is an explicit, typed, tested component; retrieval quality is
measured against labeled queries; and the cost/latency of each stage is
reported.

## 2. Goals and non-goals

Goals
- Ingest heterogeneous documents (Markdown, HTML, plain text, PDF) into a
  normalized document model with extracted metadata.
- Configurable chunking (fixed window, recursive, structure-aware) where every
  chunk carries exact character offsets, heading path, page, and source metadata.
- Hybrid retrieval: dense vector search plus BM25, fused with Reciprocal Rank
  Fusion (RRF) or weighted score fusion, with metadata filters.
- Cross-encoder reranking as an optional second stage.
- Answers that are structured (JSON schema) and cite the chunks they rely on.
- A FastAPI service exposing search, answer, and health endpoints over a
  prebuilt index (indexing stays a batch CLI job; ADR-0007).
- A reproducible evaluation harness reporting Recall@K, MRR@K, Hit Rate@K,
  nDCG@10, and p50/p95 latency per mode, run against a public labeled dataset.
- Runs end to end on a CPU laptop with zero paid API calls.

Non-goals
- Training or fine-tuning embedding models (a later portfolio project).
- Multi-tenant auth, billing, or a web UI.
- Billion-scale ANN indexing; the target scale is up to ~100k chunks.

## 3. Users and use cases

- An engineer indexing internal docs (Markdown/HTML/PDF) who needs cited answers.
- An engineer choosing between chunking/retrieval/reranking settings who needs
  measured trade-offs instead of defaults copied from a tutorial.

## 4. Architecture

```
            +-------------+     +-----------+     +------------------+
 files ---> |  Loaders    | --> | Chunkers  | --> | Embedder (local) |
            | md/html/txt |     | fixed /   |     +---------+--------+
            | pdf         |     | recursive |               |
            +------+------+     | structure |               v
                   |            +-----+-----+     +------------------+
            Document model            |           |  Vector index    |
            (sections, metadata)      +---------> |  BM25 index      |
                                                  +---------+--------+
                                                            |
 query ---> retrieve (dense + BM25) -> fuse (RRF) -> rerank (cross-encoder)
                                                            |
                                         answer (LLM provider | extractive)
                                                            |
                                  structured response with citations
```

Key interfaces (Python `Protocol`s) keep each stage swappable and testable:
`Loader`, `Chunker`, `Embedder`, `VectorStore`, `LexicalIndex`, `Reranker`,
`AnswerGenerator`. The CLI, API, and eval runner all compose the same pipeline.

## 5. Technology choices

| Concern | Choice | Rationale (see `docs/adr/`) |
| --- | --- | --- |
| Language | Python 3.11+ | typing (`StrEnum`, `Self`), ecosystem |
| Data models | pydantic v2 | validation, JSON schema for structured outputs |
| PDF parsing | pypdf | pure Python, permissive license |
| HTML parsing | stdlib `html.parser` | no extra dependency for a simple block extractor |
| Embeddings | local ONNX model via fastembed (`BAAI/bge-small-en-v1.5`) | CPU, no API spend (ADR-0002) |
| Vector store | in-process NumPy exact index behind `VectorStore` | zero infra, exact results for evals (ADR-0001) |
| Lexical search | BM25 implemented in-repo | transparent scoring, no extra dependency |
| Reranker | local cross-encoder (MiniLM MS MARCO family) | measurable quality/latency trade-off |
| LLM | OpenAI-compatible client, optional | extractive fallback keeps zero-cost default (ADR-0004) |
| API | FastAPI + uvicorn | typed request/response models, OpenAPI |
| Quality | ruff, mypy --strict, pytest | enforced in CI |
| Packaging | Docker, GitHub Actions | build validated in CI |

## 6. Data

- Development fixtures: small hand-written Markdown/HTML/text/PDF files under
  `tests/fixtures/` and generated in tests, used to verify parsing and offsets.
- Retrieval evaluation: **SciFact** via the BEIR distribution
  (5,183 scientific abstracts; 300 test queries with relevance judgments).
  Licenses: claims/annotations CC BY 4.0, abstracts ODC-By 1.0 (from S2ORC);
  see <https://github.com/allenai/scifact#license>. Data is downloaded by a
  script at evaluation time and never committed.
- A second, structure-heavy evaluation set (multi-section Markdown docs with
  hand-written queries) is planned for M4 to evaluate structure-aware chunking,
  which SciFact's single-abstract documents cannot exercise.

## 7. Evaluation methodology

- Metrics: Recall@{1,5,10}, Hit Rate@{1,5,10}, MRR@10, nDCG@10 (graded by qrels),
  plus p50/p95 retrieval latency per mode. Each timing is one query end to end
  (query embedding, search, fusion, reranking and any over-fetch), after one
  untimed warm-up per mode. Per-stage timings and index build time are not
  recorded by the harness.
- Configurations compared: BM25 only, dense only, hybrid (RRF), hybrid + rerank;
  chunk size/overlap sweeps where documents are long enough to be chunked.
- Chunk-to-document mapping: a query is credited when a retrieved chunk belongs
  to a relevant document (deduplicated by document before computing ranks).
- Every run writes a JSON report (config, dataset version, git SHA, hardware
  note, metrics) under `evals/results/`; README numbers are copied only from
  committed reports.
- Answers (M5) are graded by code, not by an LLM judge. The recorded run
  (`evals/results/answers_scifact_test_cline-pass_deepseek-v4.1-flash_20261013T013347Z.json`,
  2026-10-13, `git_sha` `c55a2bb`, `git_dirty: false`) asked
  `cline-pass/deepseek-v4.1-flash` (temperature 0) about 30 SciFact test claims
  picked with a fixed seed (0): 20 answerable claims, each with its top 5 hybrid
  (RRF) contexts, and 10 "no relevant" claims whose qrels-gold abstracts were
  removed from the 5 contexts. From the raw model JSON, code measures: replies
  parsed; citation span validity (the quote is a verbatim substring of the cited
  context and at least 20 characters); citation recall and precision against the
  qrels gold documents; false abstention on answerable claims with gold in
  context; and correct abstention when gold was removed (a lower bound, since
  qrels are incomplete). The README shows each rate with its raw count and a 95%
  Wilson interval. Every reply is cached, so `scripts/answer_eval.py --replay`
  recomputes the grades offline. Free-text faithfulness beyond the verbatim
  quotes is not measured, and no numbers are published without an executed run.

## 8. Success criteria

- Ingestion: all four formats parsed; for every chunk,
  `document.text[start:end] == chunk.text` (tested invariant).
- Retrieval: hybrid (or hybrid + rerank) beats the better of BM25-only and
  dense-only on SciFact nDCG@10, or the README explains why it does not.
- Latency: p95 search latency (without reranking) under 100 ms on the SciFact
  eval corpus (5,183 documents, 5,226 chunks), as measured by the eval harness on
  the machine its report records; reranking cost reported separately. (The
  original wording said "on a CI-class CPU", but CI runs no models, so the
  measurement comes from the committed fastembed report instead. Status: met at
  `99f28f6`, see the README latency table.)
- Engineering: CI green on lint, strict typing, tests; Docker image builds in CI;
  API has request/response schemas and error-path tests.
- Documentation: architecture, ADRs, benchmark tables generated from reports.

## 9. Milestones

See `PLAN.md` for scope and acceptance criteria of each milestone:
M1 ingestion + chunking, M2 indexing + dense/lexical retrieval,
M3 hybrid fusion + reranking, M4 evaluation harness + benchmarks,
M5 answer generation with citations, M6 FastAPI service + Docker,
M7 hardening, docs, and release.

## 10. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| No paid LLM key | Extractive answer path; LLM behind provider interface; generation evals marked not run |
| Small CPU box / CI time | small ONNX models, cached downloads, `slow` marker excludes model tests from default CI |
| PDF text extraction quality varies | per-page sections, de-hyphenation, documented limitations |
| SciFact docs are single abstracts | add a structure-heavy eval set for chunking experiments |
| Metric bugs | metric functions unit-tested against hand-computed examples |
| Benchmark numbers drifting from code | reports include git SHA + config; README tables generated from reports |

## 11. Deployment

Container image (multi-stage, non-root) running uvicorn; configuration via
`RAG_*` environment variables; the index is built by `rag-engine index` into a
volume and mounted read-only by the service (ADR-0007). The image build and an
end-to-end smoke test (`scripts/docker_smoke.sh`) run in GitHub Actions.
