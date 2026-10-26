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
- [x] Optional: full SciFact run with fastembed (±rerank) — report `evals/results/scifact_test_fastembed_20261010T184148Z.json` (git_dirty false, mean_depth_used 50). README table landed after Chad match (Oct 11).

## Review of M4 at 96b3c33 (done, see REVIEW.md)
- [x] Equal retrieval depth across modes with over-fetch to `top_k` unique docs; depth recorded in reports
- [x] Untimed warm-up per mode; `git_dirty` in reports; all-non-relevant queries skipped and counted
- [x] Structured set labeled smoke-only in reports and docs
- [x] Coverage back above 96%

## Review of 84bce05 (done, see REVIEW.md)
- [x] Corpus-scoped document ids (`corpus_id` + relative path) with pinned-id, stability and no-collision tests
- [x] `documents.jsonl` beside chunk files and in the index; citations verified on build and load; index format 2
- [x] `content_hash` and `page_end` in provenance; chunk offset validation; NaN/inf weights rejected
- [x] Per-stage `components`/`ranks` (nested rerank keeps every stage); `schema_version` on `search --json`
- [x] End-to-end offset-invariant tests across loaders and pipeline stages

## M5 - Answer generation (done)
- [x] `AnswerGenerator` protocol and pydantic `AnswerResponse` / `Citation` schema
- [x] Extractive generator (sentence selection + citations) as default
- [x] OpenAI-compatible client with JSON repair (env-configured; stdlib urllib)
- [x] `LLMConfig` / `RAG_LLM_*` settings
- [x] `rag-engine answer` CLI (`--generator extractive|openai|auto`)
- [x] Tests + coverage back at/above 96%

## Pre-M6 eval honesty pack (done 2026-10-12)
- [x] README SciFact quality table from committed fastembed report (Chad-matched); hybrid vs dense "about the same" until bootstrap CI on a sidecar-bearing run
- [x] Eval runner writes `*.per_query.json` sidecar alongside aggregate reports
- [x] `scripts/bootstrap_ci.py` + `rag_engine.eval.bootstrap` paired bootstrap CI with unit tests
- [x] Sidecar-bearing SciFact fastembed re-run at `99f28f6` committed with bootstrap CI summary; README cites the CIs (hybrid vs dense still "about the same") (2026-10-13)

## M5 answer eval (done 2026-10-13)
- [x] Code-first grader: citation span validity, citation recall/precision vs qrels gold, abstention (gold-removal ablation)
- [x] Fixed-seed subset selection recorded in reports; request-keyed JSONL reply cache with record/replay
- [x] Recorded run with `cline-pass/deepseek-v4.1-flash` (30 claims) committed with cache; README section with counts and caveats
- [ ] Optional: secondary DeepSeek judge score, labelled self-graded (not started)
- [ ] Optional: extractive-generator baseline on the same subset for comparison

## Review of M5 answer eval at 67eecad (done 2026-10-18)
- [x] 20-character minimum for valid citation quotes (`MIN_QUOTE_CHARS`) with boundary tests
- [x] README: correct abstention 8/10 described as a lower bound (incomplete qrels)
- [x] README: citation recall counts only the 17 answerable claims with gold retrieved
- [x] Every README rate shows raw count + 95% Wilson interval (`scripts/answer_eval_table.py`)
- [x] Offline cache replay under the new rule: every per-item grade unchanged (2026-10-18)

## M6 - FastAPI service (done 2026-10-20, ADR-0007)
- [x] `GET /health`, `POST /search`, `POST /answer` over a prebuilt read-only index (`RAG_INDEX_DIR`)
- [x] Request schemas with the CLI's option rules; 422/400/503/502 error contract; injected reranker/generator factories
- [x] Search hits match `rag-engine search --json` (parity test per mode)
- [x] Dockerfile (multi-stage, non-root, HEALTHCHECK), `.dockerignore`, `compose.yaml`
- [x] CI `docker` job: image build, compose config check, `scripts/docker_smoke.sh`
- [x] `POST /ingest` dropped from M6: indexing stays a batch job (ADR-0007)

## M7 - Hardening and release (in progress)
- [x] Locked first model load in the fastembed embedder/reranker + opt-in `RAG_WARMUP=off|embedder|all` (2026-10-25)
- [x] JSON request log line per request (request id, status, `duration_ms`, `retrieve_ms`, `generate_ms`; no query text) + `X-Request-ID`; `RAG_LOG_LEVEL` (2026-10-25)
- [x] compose binds `127.0.0.1:8000:8000`; README/Dockerfile say there is no auth (2026-10-25)
- [ ] Pin the Docker base image by digest
- [x] CI: `scripts/compose_smoke.sh` builds a tiny hashing index with the compose `index` service, then `docker compose up -d --wait api`, the shared smoke checks (`scripts/smoke_lib.sh`) and `docker compose down -v`; README now says compose is tested (2026-10-26, green at 17c3561)
- [ ] Extractive answers: the first sentence of a chunk can include its Markdown heading lines (seen in the Docker smoke test); strip headings from extracted sentences
- [ ] v1.0.0 release notes, Chad's final REVIEW.md
