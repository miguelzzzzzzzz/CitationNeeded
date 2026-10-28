# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `scripts/eval_table.py --report <file>` renders one report in detail: nDCG@10, MRR@10, Recall@{1,5,10} and Hit@{1,5,10} per mode, p50/p95 retrieval latency per mode (`n/a` where the report has no timing), and the report's hardware record verbatim. The README's SciFact tables come from its output.
- Service hardening (M7): opt-in startup warm-up `RAG_WARMUP=off|embedder|all` (`RagService.warm_up`, run before the first request is accepted; a failed warm-up is logged and the model loads on first use); one JSON log line per request on `rag_engine.api.requests` (request id, method, path, status, `duration_ms`, `retrieve_ms`, `generate_ms`; no query text; `/health` at DEBUG) and an `X-Request-ID` response header; `RAG_LOG_LEVEL`. The Docker smoke test checks the warm-up and request log lines.

### Changed
- Docs (Chad's final review at `900575f`): the README adds Recall/Hit@1 and @5 to the SciFact quality table, a per-mode p50/p95 latency table from the `99f28f6` report with its recorded hardware, and a known-limitations section (including that heading stripping in extractive answers cannot see a fence opened before the chunk starts). "Per-stage latency" now reads "latency per mode", since the harness times each query end to end. PROJECT_SPEC §7 describes the code-graded M5 answer run as it was run; §8's p95 < 100 ms target (without reranking) is now measured on the machine the report records instead of "a CI-class CPU" (CI runs no models), and it is met: hybrid p95 61.4 ms, dense 40.1 ms, lexical 7.7 ms.
- The Dockerfile pins its base image by the digest of the `python:3.13-slim` multi-arch index (`PYTHON_IMAGE` build arg, resolved 2026-10-27) for both stages, so rebuilds use the same OS and Python layers; the Dockerfile says how to refresh it.
- The CI `docker` job also runs `scripts/compose_smoke.sh`: `docker compose run --rm index` builds a hashing-embedder index of `evals/structured/docs` (no model downloads) before `docker compose up -d --wait api` (the healthcheck is 503 until an index is served), checks that the API is published on `127.0.0.1:8000` with a read-only root filesystem and a read-only index mount, runs the same HTTP checks as the `docker run` smoke test (now shared in `scripts/smoke_lib.sh`), then `docker compose down -v`. `compose.yaml` takes the image name from `RAG_IMAGE` (default `citation-needed:local`).
- The fastembed embedder and cross-encoder serialize their lazy first model load with a lock, so concurrent first requests construct one model instead of one each.
- `compose.yaml` publishes the API on `127.0.0.1:8000` only, and the README and Dockerfile state that the service has no authentication. The image runs uvicorn with `--no-access-log` (the request log lines replace it).

### Fixed
- Extractive answers no longer quote Markdown heading lines: for Markdown and HTML chunks (HTML headings are normalized to ATX lines), ATX heading lines are sentence boundaries and are dropped, so a chunk's first sentence is no longer `# Title\n\n## Section\n\nText...`. Sentence spans are now computed directly instead of re-found with `str.find`. Plain-text and PDF chunks are unchanged. The Docker and compose smoke tests check that no answer or quote contains a heading line.
- Under uvicorn the service's own INFO log lines (e.g. "serving index") were dropped because nothing configured a handler for `rag_engine`; the app now adds a stderr handler when neither the root logger nor `rag_engine` has one.

### Added
- HTTP service (M6, ADR-0007): `rag_engine.api` with `GET /health`, `POST /search` and `POST /answer` (FastAPI, optional extra `api`; run `uvicorn rag_engine.api.app:create_app --factory`). Serves one prebuilt index read-only from `RAG_INDEX_DIR`; `/health` is 200 only while an index is served (503 with the reason otherwise), and a corrupt index stops startup. Request schemas reject unknown fields and enforce the CLI's option rules; errors map to 422/400/503/502, and a `ValueError` during retrieval stays a 500. `rerank: true` uses only the server's `RAG_RERANK_MODEL`. The framework-free `RagService` takes injectable reranker and generator factories, so tests use fakes and need no model. Validation errors do not echo the rejected input back (NaN/Infinity would make the error body invalid JSON).
- `Dockerfile` (multi-stage, `python:3.13-slim`, non-root uid 10001, `HEALTHCHECK` on `/health`), `.dockerignore`, `compose.yaml` (`index` tool service + read-only `api`), and a CI `docker` job that builds the image, validates the compose file and runs `scripts/docker_smoke.sh` (hashing index built in the container, served from a read-only mount, every endpoint exercised, 503 and a failing `HEALTHCHECK` without an index).
- `ServiceConfig` in `Settings` (`RAG_INDEX_DIR`, `RAG_RERANK_MODEL`).

### Changed
- `rag_engine.retrieval.compose_retriever` and `rag_engine.answering.select_generator` now build the retriever stack and pick the answer generator for both the CLI and the service (previously duplicated in `cli.py`). The `openai`-without-settings error now reads "(or use the extractive generator)".
- `.env.example` documents the LLM settings as live (they have been read since M5) and the new service settings.
- Answer eval (Chad's M5 review MINORs): a raw citation quote must be at least `MIN_QUOTE_CHARS` (20) characters after trimming whitespace to count as a valid span; `rag_engine.eval.intervals.wilson_interval` and `scripts/answer_eval_table.py` render every README rate with its raw count and a 95% Wilson interval, recomputed from per-item grades and checked against the report's metrics; README states that correct abstention (8/10) is a lower bound and that citation recall counts only the 17 answerable claims with gold retrieved. An offline replay of the committed cache under the new rule (2026-10-18, 30 cache hits, 0 API calls) reproduced every per-item grade, so the published numbers are unchanged.

### Added
- First recorded M5 answer eval (`evals/results/answers_scifact_test_cline-pass_deepseek-v4.1-flash_20261013T013347Z.json`, cache under `evals/answers/`): 30 seeded SciFact test claims, `cline-pass/deepseek-v4.1-flash`, run 2026-10-13 at `c55a2bb`. Code-graded: span validity 29/30, citation recall 0.71 (17 eligible), precision 18/29, false abstention 4/17, correct abstention 8/10. README section with small-sample caveats.
- M5 answer eval (code-graded): `rag_engine.eval.answers` (fixed-seed subset selection, raw-reply parsing, citation span validity, citation recall/precision vs qrels gold, abstention grading, aggregation), `rag_engine.eval.answer_cache` (request-keyed JSONL reply cache and `CachedAnswerGenerator` with record/replay modes), `rag_engine.eval.answer_runner` (gold-removal `no_relevant` items with exactly k contexts, run loop, report), and box-only `scripts/answer_eval.py`. The OpenAI-compatible client now unwraps `{"data": {...}}` gateway envelopes (Cline).
- SciFact fastembed re-run at `99f28f6` (`git_dirty: false`) with its per-query sidecar and `….bootstrap_ci.json` (paired bootstrap, 95%, B=10,000, seed 0). Aggregates match the 2026-10-10 report exactly. hybrid − dense nDCG@10 is −0.0015 [−0.0273, +0.0239], so the README keeps "about the same"; dense and hybrid beat lexical (CIs exclude 0); hybrid+rerank − hybrid spans 0.
- Eval honesty pack (pre-M6): `run_evaluation` returns per-query scores; `write_report` writes a sibling `*.per_query.json` sidecar (schema 1). `rag_engine.eval.bootstrap.paired_bootstrap_ci` + `scripts/bootstrap_ci.py` for paired hybrid-vs-dense CIs. README SciFact quality table from the committed fastembed report (Chad-matched aggregates; hybrid vs dense described as about the same until a sidecar-bearing bootstrap CI is cited).
- Answer generation (M5 / ADR-0004): `rag_engine.answering` with `AnswerGenerator`
  protocol, pydantic `AnswerResponse`/`Citation` schema, extractive default
  (`ExtractiveAnswerGenerator`), and optional OpenAI-compatible client
  (`OpenAICompatibleAnswerGenerator` via stdlib urllib; configured by
  `RAG_LLM_BASE_URL`, `RAG_LLM_MODEL`, `RAG_LLM_API_KEY`). `rag-engine answer`
  retrieves then generates cited answers (`--generator extractive|openai|auto`).

### Fixed
- Evaluation (M4 review): every mode is compared at the same retrieval depth. Each starts at `EvalConfig.depth` (default 50, at least `top_k`) and over-fetches until it has `top_k` unique documents or the index or `max_depth` is exhausted. Previously dense/lexical fetched only `top_k` chunks while hybrid fetched 50, so baselines could be scored on fewer documents. Reports record per-mode depth statistics and `queries_short_of_k`; `rag-engine evaluate` gains `--depth`/`--max-depth`.
- Evaluation reports: one untimed warm-up per mode before latency timing; `git_dirty` recorded with `git_sha`; queries with no relevant grade are skipped and counted (`n_queries_skipped_no_relevant`, `n_queries_unlabelled`) instead of scored 0; structured-set reports are flagged `smoke_only`; report `schema_version` 2.

### Added
- Evaluation harness (M4): `rag_engine.eval.metrics` (Recall@{1,5,10}, Hit@{1,5,10}, MRR@10, nDCG@10) with hand-computed unit tests; BEIR SciFact loader with MD5 verification (`evals/datasets/` gitignored); in-repo structured-doc set under `evals/structured/`; `run_evaluation` / `run_scifact_eval` / `run_structured_eval` writing JSON reports under `evals/results/`; `rag-engine evaluate` CLI; `scripts/eval_table.py` for README tables. Structured smoke report committed (hashing embedder — not a retrieval-quality claim).

## [0.1.0] - 2026-10-09

First tagged release: a pre-1.0 **interface release** covering milestones M1-M3
(ingestion and chunking, indexing and retrieval, hybrid fusion and reranking). It
passed the interface review ([REVIEW.md](REVIEW.md)). No evaluation numbers exist
yet; retrieval quality is measured in M4. Interfaces may still change before 1.0.

**Breaking changes** (relative to earlier untagged builds): document ids are
corpus-scoped (`sha256("<corpus_id>:<source>")[:16]`), index format 2 stores
`documents.jsonl` and rejects format-1 indexes (rebuild chunk files and indexes),
`save_index` takes the documents, `search --json` returns a versioned object
(`schema_version` 1) with the hits under `hits`, and reranking reports the base
stage's rank in `ranks` instead of a `base_rank` score key. Always pass
`--corpus-id`; the default falls back to the folder name.

### Changed
- **Breaking:** document ids are now `sha256("<corpus_id>:<source>")[:16]`. The corpus id is set with `--corpus-id`, `RAG_CORPUS_ID` or `IngestionConfig.corpus_id`, and defaults to the slugified corpus root directory name. Ids no longer change with the ingest root, and they no longer collide across corpora. Chunk files and indexes built before this change must be rebuilt (review MAJOR 2; ADR-0006).
- **Breaking:** index format 2 stores `documents.jsonl` (the normalized documents that chunk offsets point into). `load_index` rejects format 1 and asks for a rebuild. `save_index` takes the documents as a new argument (review MAJOR 1).
- **Breaking:** `search --json` prints `{"schema_version": 1, "query", "mode", "rerank", "hits": [...]}` instead of a bare list.
- Reranking records the base stage's score in `components` and its rank in the new `RetrievedChunk.ranks`, both keyed by stage name. The `base_rank` score key is gone, and nested reranks keep every stage.
- Renamed the repository to `CitationNeeded` (display title "Citation Needed"). The distribution, import package, and CLI names are unchanged.

### Added
- `rag_engine.documents` (`write_documents`, `read_documents`, `verify_chunks`). `rag-engine ingest` writes `<stem>.documents.jsonl` next to the chunks. `index`, `save_index` and `load_index` verify that every chunk's text equals `document.text[start_char:end_char]` and that `content_hash` and `source` match. `load_documents` returns the normalized text behind an index.
- `provenance()` includes `content_hash` and `page_end`.
- End-to-end offset-invariant tests: every loader through chunking, save/load, RRF, weighted fusion, rerank and nested rerank.
- `REVIEW.md` records Chad's review at `84bce05` and how each finding was resolved.
- Hybrid retrieval (M3): `reciprocal_rank_fusion` (weighted RRF, k = 60 by default) and `weighted_score_fusion` (per-list, per-query min-max normalization), deterministic tie-breaking, and `HybridRetriever`, which queries each retriever for a candidate pool with the same metadata filters. Fused hits record their per-retriever scores in `RetrievedChunk.components`.
- Reranking (M3): `Reranker` protocol, `CrossEncoderReranker` (fastembed `TextCrossEncoder`, default `Xenova/ms-marco-MiniLM-L-6-v2`, lazy optional import), and `RerankingRetriever` with a candidate budget. Real-model tests are marked `slow`.
- `rag-engine search` gains `--mode hybrid`, `--fusion rrf|weighted`, `--weight NAME=FLOAT`, `--rrf-k`, `--candidates`, `--rerank`, and `--rerank-model`; human-readable output shows component scores.
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
- `Chunk` validates `end_char > start_char` and `len(text) == end_char - start_char`. `Document` validates its `doc_id` and `content_hash`.
- Fusion weights that are NaN or infinite are rejected.
- Front-matter keys can no longer shadow `source`, `corpus_id` or `content_hash` in chunk metadata.
- Index persistence (M2 review): `save_index` refuses embedders whose name is not a reloadable spec and retrievers whose chunk-id sets differ, and removes an old manifest before rewriting; `load_index` validates manifest fields, checks that both halves hold the same chunk ids, accepts the build-time BM25 tokenizer, and passes the manifest dimension to the embedder so lexical search on a fastembed-built index no longer needs fastembed.
- CLI (M2 review): a missing fastembed install is reported as `error: ...` (exit 2) instead of a traceback; chunking options passed with a chunks `.jsonl` input produce a warning since they have no effect.
- Chunking overrides on the command line no longer reset `RAG_EMBEDDING_*` (and other non-chunking) settings to defaults.
- Dense retrieval returns no hits for a query whose embedding is all zeros instead of an arbitrary list of zero-score chunks.

[Unreleased]: https://github.com/miguelzzzzzzzz/CitationNeeded/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/miguelzzzzzzzz/CitationNeeded/releases/tag/v0.1.0
