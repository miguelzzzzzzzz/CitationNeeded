# Citation Needed

A production-style hybrid RAG service that answers with citations to exact source spans.

[![CI](https://github.com/miguelzzzzzzzz/CitationNeeded/actions/workflows/ci.yml/badge.svg)](https://github.com/miguelzzzzzzzz/CitationNeeded/actions/workflows/ci.yml)

A retrieval-augmented generation engine built the way a production team would
build one: typed, swappable pipeline stages; structure-aware chunking with
exact character offsets into the normalized document text (stored with the
index, so every citation can be re-checked); hybrid dense + BM25 retrieval with reranking; cited,
schema-validated answers; and a reproducible evaluation harness so every
design choice is backed by measured retrieval quality and latency.

> **Status:** v0.1.0, a pre-1.0 interface release (see [PLAN.md](PLAN.md) and
> [CHANGELOG.md](CHANGELOG.md)). M1–M6 (ingestion through cited answer
> generation and the HTTP service with its Docker image) are done; M1–M3 passed
> interface review ([REVIEW.md](REVIEW.md)).
> Interfaces may still change before 1.0. SciFact quality numbers below come
> from the committed fastembed report; hashing / structured reports under
> `evals/results/` are harness/smoke checks only. Regenerate tables with
> `python scripts/eval_table.py` (one report in detail: `--report <file>`) — do
> not edit numbers by hand.

## Why this is not "chat with a PDF"

| Typical demo | This project |
| --- | --- |
| One loader, fixed-size chunks | Markdown/HTML/text/PDF loaders, three chunking strategies, heading-path and page metadata |
| Top-k cosine only | Dense + BM25 hybrid with rank fusion, metadata filters, cross-encoder reranking |
| Free-text answer | Pydantic-validated response with citations to exact character spans |
| "It looks right" | Recall@K, MRR, Hit Rate, nDCG and p50/p95 latency per mode on SciFact (BEIR) |
| Requires a paid API | Runs fully on CPU with local ONNX models; LLM is optional behind a provider interface |

## Architecture

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full design and
[docs/adr/](docs/adr/) for the decisions behind it.

```
files -> loaders -> Document(sections, metadata) -> chunkers -> embedder -> vector index + BM25
query -> dense + BM25 -> fusion -> reranker -> answer generator (LLM | extractive) -> cited JSON
```

## Usage (current milestone)

```bash
pip install -e .
rag-engine ingest path/to/docs --corpus-id handbook --out chunks.jsonl --strategy structure --chunk-size 256 --chunk-overlap 32
```

This loads Markdown, HTML, PDF, and text files, writes one JSON chunk per line
(text, exact character span, heading path, page, metadata), and prints chunk
statistics plus any skipped files with the reason. Details:
[docs/ingestion-and-chunking.md](docs/ingestion-and-chunking.md).

Offsets index the *normalized* document text (NFKC, so e.g. the `ﬁ` ligature
becomes `fi`; CRLF becomes LF; control characters are removed), not raw-file
bytes. `ingest` therefore also writes `chunks.documents.jsonl` next to
`chunks.jsonl`, and every index directory carries `documents.jsonl`: the
normalized text and its sha256 `content_hash` for each document. `index` and
`load_index` re-check every chunk against it (`chunk.text ==
document.text[start_char:end_char]`, matching `content_hash`) and refuse
anything that does not match.

Document ids are `sha256("<corpus_id>:<relative/posix/path>")[:16]`, so the same
corpus gives the same ids wherever it is checked out, and two corpora never share
ids. `corpus_id` defaults to the corpus root directory name (slugified) and can be
set with `--corpus-id`, `RAG_CORPUS_ID`, or `IngestionConfig.corpus_id`; see
[ADR-0006](docs/adr/0006-corpus-scoped-ids-and-verifiable-offsets.md). Index
format 2 introduced both changes; format-1 indexes are rejected with a request
to rebuild.

**Always pass `--corpus-id`** (or set `RAG_CORPUS_ID`) when ingesting or
indexing raw documents. Without it the corpus id falls back to the folder name,
so renaming the folder changes every document id, and two different corpora
that happen to live in folders with the same name (for example `docs/`) get
the same corpus id and can collide. Pick a short, stable name per corpus
(`^[a-z0-9][a-z0-9._-]{0,63}$`) and keep it for the corpus's lifetime.

Build an index directory (dense vectors + BM25) and query it:

```bash
rag-engine index path/to/docs --corpus-id handbook --out index/   # or a chunks.jsonl from `ingest`
rag-engine search index/ "how does reranking work" -k 5            # dense (bge-small)
rag-engine search index/ "BM25 k1" --mode lexical --filter source=guide.md --json
```

Hybrid retrieval and reranking:

```bash
rag-engine search index/ "how does reranking work" --mode hybrid                  # RRF (k=60)
rag-engine search index/ "BM25 k1" --mode hybrid --fusion weighted --weight lexical=2
rag-engine search index/ "how does reranking work" --mode hybrid --rerank --candidates 30
```

Each hit carries its rank, score, retriever, provenance (document and chunk id,
source file, `content_hash`, heading path, `page`/`page_end`, character span),
and how it got there: `components` maps each upstream stage name to the score it
gave the chunk and `ranks` maps the same stage names to the 1-based rank it had
there (e.g. `{"dense": 2, "lexical": 1, "hybrid": 3}` after reranking a hybrid
search). A wrapper stage's name extends its base's (`hybrid` -> `hybrid+rerank`),
so nested stages never overwrite each other. Metadata filters are applied inside
every retriever before ranking.

`search --json` prints a versioned object, documented here as schema version 1:

```json
{"schema_version": 1, "query": "...", "mode": "hybrid", "rerank": true,
 "hits": [{"rank": 1, "score": 0.93, "retriever": "hybrid+rerank",
           "components": {...}, "ranks": {...}, "doc_id": "...", "chunk_id": "...",
           "source": "...", "content_hash": "...", "heading_path": [...],
           "page": null, "page_end": null, "start_char": 0, "end_char": 120,
           "text": "..."}]}
```

`components`/`ranks` are omitted for a single retriever. Any breaking change to
these keys bumps `schema_version`.
`--rerank` scores at most `--candidates` (query, chunk) pairs with the
`Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder (~80 MB, CPU). `--embedder hashing`
builds a fully offline index for testing. Which configuration retrieves best is
measured by the M4 harness (`rag-engine evaluate`); see the SciFact table below.

## HTTP API

A FastAPI service serves one prebuilt index read-only
([ADR-0007](docs/adr/0007-read-only-http-service.md)): build the index with
`rag-engine index`, point `RAG_INDEX_DIR` at it, and start uvicorn. Rebuilding
the index means restarting the service.

```bash
pip install -e ".[api,embeddings]"
rag-engine index path/to/docs --corpus-id handbook --out index/
RAG_INDEX_DIR=index uvicorn rag_engine.api.app:create_app --factory --port 8000
```

| Endpoint | Body | Returns |
| --- | --- | --- |
| `GET /health` | | 200 `{"status": "ok", "index_loaded": true, "chunks", "documents", "embedder", ...}` while an index is served; 503 with `detail` otherwise |
| `POST /search` | `{"query", "k": 5, "mode": "hybrid", "fusion", "rrf_k", "weights", "candidates", "rerank", "filters"}` | `{"schema_version": 1, "query", "mode", "rerank", "hits": [...]}` |
| `POST /answer` | the `/search` fields plus `"generator": "auto"`, `"max_sentences": 5` | the `AnswerResponse` JSON: `answer`, `citations` (doc/chunk id, character span, quote), `abstained`, `generator` |

```bash
curl -s localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "how does reranking work", "k": 3, "mode": "hybrid", "filters": {"source": "guide.md"}}'
curl -s localhost:8000/answer -H 'Content-Type: application/json' \
  -d '{"query": "how does reranking work", "generator": "extractive"}'
```

Search hits have the same fields as `search --json` hits, except that
`components` and `ranks` are always present (empty for a single retriever).
Options follow the CLI's rules: `fusion`/`rrf_k`/`weights` need `mode:
"hybrid"`, `candidates` needs hybrid or `rerank`, and `filters` maps a field to
a value or an "any of" list. `rerank: true` uses the server's cross-encoder
(`RAG_RERANK_MODEL`); clients cannot pick a model. `generator: "auto"` uses the
OpenAI-compatible client when `RAG_LLM_BASE_URL` and `RAG_LLM_MODEL` are set and
the extractive generator otherwise. Errors return `{"detail": ...}`: 422 for an
invalid body, 400 for a valid request the server cannot serve (e.g.
`"generator": "openai"` without `RAG_LLM_*`), 503 when no index is served or an
optional dependency (fastembed) is missing, 502 when the answer generator fails.
OpenAPI docs are at `/docs`.

**There is no authentication.** Anyone who can reach the port can query the
index and, with `RAG_LLM_*` set, spend LLM calls. Keep it on localhost or a
trusted network, or put an authenticating reverse proxy in front of it;
`compose.yaml` publishes the port on `127.0.0.1` only.

Models load lazily on the first request that needs them; the fastembed adapters
serialize that first load, so concurrent first requests share one model.
`RAG_WARMUP=embedder` embeds one probe query at startup (and `all` also scores
one pair with the reranker, downloading it if needed), so the first real
request does not pay the load and the Docker `HEALTHCHECK` turns healthy only
once the models are warm. A warm-up that fails (e.g. no network for the
download) is logged and the model loads on first use instead.

Each response carries an `X-Request-ID` header, and each request is logged as
one JSON line on the `rag_engine.api.requests` logger (`RAG_LOG_LEVEL`, default
`INFO`; `/health` at `DEBUG`):

```text
INFO rag_engine.api.requests {"duration_ms": 2.353, "event": "request", "method": "POST", "path": "/search", "request_id": "84042302d3e34cd09e50a6b95354f53a", "retrieve_ms": 0.062, "status": 200}
```

`retrieve_ms` covers retrieval (fusion and reranking included) and
`generate_ms` the answer generator; query text is never logged. The Docker image
turns off uvicorn's access log in favour of these lines.

### Docker

The image (`python:3.13-slim` pinned by digest in the [Dockerfile](Dockerfile),
multi-stage, runs as uid 10001) serves the index
mounted at `/data/index` and caches fastembed models under `/data/models`.
Models download on first use, not at build time. With
[compose.yaml](compose.yaml), put documents in `./corpus` and run:

```bash
docker compose run --rm index      # rag-engine index /corpus -> rag-index volume (fastembed)
docker compose up -d api           # serves it read-only on 127.0.0.1:8000, read-only root filesystem
```

CI builds the image and runs [scripts/docker_smoke.sh](scripts/docker_smoke.sh):
it builds a hashing-embedder index of `evals/structured/docs` inside the
container, serves it from a read-only mount, checks every endpoint (dense,
lexical and hybrid search, an extractive answer with citations, 422/400 errors),
checks the startup warm-up and the JSON request log line (with no query text),
and checks that a container without an index answers 503 and fails its
`HEALTHCHECK`. CI also runs [scripts/compose_smoke.sh](scripts/compose_smoke.sh)
on the same image: `docker compose run --rm index` builds the hashing index
first (the healthcheck stays 503 until an index is served), then
`docker compose up -d --wait api` must turn healthy; the script checks the
`127.0.0.1:8000` binding, the read-only root filesystem and index mount, runs
the same endpoint and log checks, and ends with `docker compose down -v`. No
model is downloaded in CI, so the fastembed default of the `index` service is
not exercised there (`RAG_INDEX_EMBEDDER=hashing`; `RAG_IMAGE` picks the image).

## Naming

The repository is called *Citation Needed*. The Python distribution
(`production-rag-engine`), import package (`rag_engine`), and CLI
(`rag-engine`) keep their descriptive names.

## Evaluation

```bash
# Structured-doc smoke (in-repo fixtures; hashing embedder — harness check only)
rag-engine evaluate --dataset structured --embedder hashing

# SciFact (downloads into evals/datasets/, gitignored; MD5-verified)
rag-engine evaluate --dataset scifact --embedder hashing
# For real dense quality: pip install -e ".[embeddings]" and --embedder fastembed
```

Every mode is compared at the same retrieval depth (`--depth`, default 50
chunks), and each query over-fetches until it has `--top-k` unique documents.
Before timing, each mode runs one untimed warm-up query. Reports record the
depth used, `git_sha` plus `git_dirty`, and how many queries were skipped
because they have no relevant judgement. The structured set (3 docs, 8
queries) is a smoke test: its reports carry `"smoke_only": true` and support
no quality claims.

JSON reports land in `evals/results/` (plus a `*.per_query.json` sidecar for
paired bootstrap). Render a Markdown table with `python scripts/eval_table.py`.
Details: [evals/README.md](evals/README.md).

### SciFact (BEIR test, fastembed)

Source: [`evals/results/scifact_test_fastembed_20261010T184148Z.json`](evals/results/scifact_test_fastembed_20261010T184148Z.json)
(`git_dirty: false`, depth 50, 300 scored queries, embedder `BAAI/bge-small-en-v1.5`,
fusion RRF k=60, reranker `Xenova/ms-marco-MiniLM-L-6-v2`). Chad independently
re-ran and matched these aggregates to three decimals (2026-10-11). The clean
re-run below holds identical metrics, @1/@5 included; this table is the output
of `python scripts/eval_table.py --report` on either report.

| mode | nDCG@10 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | Hit@1 | Hit@5 | Hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.7148 | 0.6806 | 0.5787 | 0.7736 | 0.8412 | 0.6033 | 0.7933 | 0.8567 |
| lexical | 0.6691 | 0.6359 | 0.5375 | 0.7259 | 0.7908 | 0.5533 | 0.7500 | 0.8100 |
| hybrid | 0.7133 | 0.6766 | 0.5673 | 0.7728 | 0.8446 | 0.5900 | 0.7933 | 0.8567 |
| hybrid+rerank | 0.6900 | 0.6550 | 0.5451 | 0.7385 | 0.8252 | 0.5700 | 0.7600 | 0.8367 |

A clean re-run at `99f28f6` (`git_dirty: false`,
[`scifact_test_fastembed_20261012T013527Z.json`](evals/results/scifact_test_fastembed_20261012T013527Z.json))
reproduced every aggregate metric above exactly and wrote a per-query sidecar. Paired
bootstrap 95% CIs on its 300 queries (B=10,000, seed 0;
[`…bootstrap_ci.json`](evals/results/scifact_test_fastembed_20261012T013527Z.bootstrap_ci.json)):

| comparison (nDCG@10) | mean diff | 95% CI |
| --- | ---: | ---: |
| hybrid − dense | −0.0015 | [−0.0273, +0.0239] |
| hybrid+rerank − hybrid | −0.0233 | [−0.0531, +0.0057] |
| dense − lexical | +0.0458 | [+0.0085, +0.0829] |
| hybrid − lexical | +0.0442 | [+0.0201, +0.0686] |

So **hybrid and dense are about the same** on this setup (the interval spans
0), and the reranker's drop versus hybrid is not distinguishable from noise
either. Dense and hybrid both beat lexical BM25 (intervals exclude 0). This is one
dataset with one embedder, so treat it as SciFact-specific.

### SciFact retrieval latency per mode

From the same clean re-run at `99f28f6`
([`scifact_test_fastembed_20261012T013527Z.json`](evals/results/scifact_test_fastembed_20261012T013527Z.json),
300 queries; `python scripts/eval_table.py --report <file>` prints both tables
above and this one):

| mode | retrieve p50 (ms) | retrieve p95 (ms) |
| --- | ---: | ---: |
| dense | 23.9 | 40.1 |
| lexical | 5.0 | 7.7 |
| hybrid | 38.1 | 61.4 |
| hybrid+rerank | 1477.7 | 1751.6 |

Hardware as recorded in the report: platform `Linux-6.12.94+-x86_64-with-glibc2.41`,
processor (empty), python `3.13.5`, cpu_count `8`. The report does not record
the CPU model, and the run was on a development machine, not a CI runner (CI
runs no models).

Each timing is one query end to end: query embedding (dense and hybrid), exact
search over 5,226 chunks, fusion, reranking of 50 candidates (hybrid+rerank),
and collapsing hits to 10 unique documents, after one untimed warm-up per mode.
The harness does not time stages separately or record index build time.

**Spec target ([PROJECT_SPEC.md](PROJECT_SPEC.md) §8): p95 search latency
without reranking under 100 ms. Met on this machine:** the slowest mode without
reranking is hybrid at a p95 of 61.4 ms. Reranking is the cost to watch: the
cross-encoder takes hybrid from a p95 of 61.4 ms to 1751.6 ms, and on SciFact
it does not improve quality (see the CIs above). These are single-process
timings on one machine, not a load test of the HTTP service.

### Answer generation on SciFact (M5, code-graded, small sample)

Source: [`evals/results/answers_scifact_test_cline-pass_deepseek-v4.1-flash_20261013T013347Z.json`](evals/results/answers_scifact_test_cline-pass_deepseek-v4.1-flash_20261013T013347Z.json)
(model `cline-pass/deepseek-v4.1-flash` via the Cline API, run 2026-10-13, `git_sha`
`c55a2bb`, `git_dirty: false`). 30 test claims picked with a fixed seed (seed 0):
20 answerable plus 10 "no relevant" claims whose qrels-gold abstracts were removed
from the 5 hybrid-retrieved contexts. Graded by code from the raw model JSON; no
LLM judge. Every reply is cached in
[`evals/answers/`](evals/answers/), and
`python scripts/answer_eval.py --replay` recomputes these numbers without the API
(a replay on the box matched every record: 30 cache hits, 0 API calls).

| metric | value | count | 95% Wilson CI |
| --- | --- | --- | --- |
| JSON replies parsed | 1.00 | 30 / 30 | [0.89, 1.00] |
| citation span validity (quote verbatim in cited context, >= 20 chars) | 0.97 | 29 / 30 | [0.83, 0.99] |
| citation recall vs gold docs in context (answerable, gold retrieved) | 0.71 | 12 / 17 | [0.47, 0.87] |
| citation precision (valid citations to a gold doc) | 0.62 | 18 / 29 | [0.44, 0.77] |
| false abstention (answerable, gold in context) | 0.24 | 4 / 17 | [0.10, 0.47] |
| correct abstention (gold removed; lower bound) | 0.80 | 8 / 10 | [0.49, 0.94] |

The table is generated by `python scripts/answer_eval_table.py <report>` from the
per-item grades in the report; each rate shows its raw count and a 95% Wilson
score interval (`rag_engine.eval.intervals.wilson_interval`). A citation counts
as valid only if its quote is a verbatim substring of the cited context and is
at least 20 characters long after trimming surrounding whitespace
(`MIN_QUOTE_CHARS`), so a short string that appears in the context by chance is
not credited. The shortest quote in this run is 75 characters, and offline
replays of the committed cache under the 20-character rule (2026-10-14 and
2026-10-18; 30 cache hits, 0 API calls) reproduced every per-item grade, so the
numbers above did not change.

How to read it:

- **Citation recall** counts only the 17 answerable claims whose gold abstract
  was among the 5 retrieved contexts. The other 3 of the 20 answerable claims
  had no gold abstract in the top 5, so the model could not cite it; they are
  excluded from recall and from false abstention. Each of the 17 had exactly one
  gold abstract in context, so recall is a 12 / 17 proportion here.
- **Read correct abstention (8/10) as a lower bound.** SciFact qrels are
  incomplete, and the only two misses, the "no relevant" claims the model
  answered (118, 1226), cite non-gold abstracts that do discuss the claim. They
  may be correct answers that the labels score as failures; the grader cannot
  tell. Precision is penalised the same way.
- This is a 30-item sample: each item moves a rate by several points, and the
  intervals are wide (e.g. correct abstention [0.49, 0.94]). Treat these as a
  sanity check of the grounding pipeline, not as a model benchmark.

## Known limitations

- **Heading lines inside fenced code blocks (extractive answers).** For Markdown
  and HTML chunks the extractive generator drops ATX heading lines from quotes.
  It finds them with the loader's fence-aware scanner, but runs it on each chunk
  on its own. A chunk that starts inside a fenced code block cannot see the
  opening fence, so `#` comment lines in that block are taken for headings and
  dropped, and the closing fence is read as an opening one, so real headings
  after it can end up in a quote. Chunks that start outside a fence are handled
  correctly.
- **One evaluation dataset.** Retrieval quality and latency are measured on
  SciFact only, with one embedder (`BAAI/bge-small-en-v1.5`), on one machine.
  The structured-doc set is a smoke test.
- **Small answer eval.** The M5 answer numbers come from 30 claims and one model
  run; read them as a sanity check (see the intervals above).
- **Exact search only.** Dense search scans every vector (NumPy), which is fine
  at SciFact scale (5,226 chunks) but grows linearly with the corpus.
- **Read-only service, no authentication.** The HTTP service serves one prebuilt
  index; rebuilding it means restarting the service, and anyone who can reach
  the port can query it (see [HTTP API](#http-api)).

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,api]"
ruff check . && ruff format --check . && mypy && pytest -m "not slow"
```

Dense embeddings use [fastembed](https://github.com/qdrant/fastembed) with
`BAAI/bge-small-en-v1.5` (384 dimensions, CPU ONNX, ~67 MB download on first
use). It is an optional extra; tests that run the real model are marked
`slow` and are not part of CI:

```bash
pip install -e ".[dev,embeddings]"
pytest -m slow
```

Configuration is read from `RAG_*` environment variables; see
[.env.example](.env.example).

## Project documents

- [PROJECT_SPEC.md](PROJECT_SPEC.md): problem, scope, architecture, evaluation, risks
- [PLAN.md](PLAN.md): milestones and acceptance criteria
- [TODO.md](TODO.md): current task list
- [CHANGELOG.md](CHANGELOG.md)
- [evals/README.md](evals/README.md): evaluation methodology and results

## License

MIT. Evaluation data (SciFact) is downloaded at run time and keeps its own
licenses (CC BY 4.0 for claims, ODC-By 1.0 for abstracts).
