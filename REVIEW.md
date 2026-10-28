# Reviews

## Chad's review at `84bce05` (2026-10-09)

Result: 0 CRITICAL, 2 MAJOR, plus MINOR and OPTIONAL items. Every finding below
is resolved; the commits are on `main`.

### MAJOR 1: the index cannot verify its citations

Chunk offsets index text that loaders have normalized (NFKC, so ligatures like
`ﬁ` become `fi`; CRLF becomes LF; control characters are removed), and that
normalized text was not saved. `provenance()` (`retriever.py:42-52`) had no
content hash, and README.md:9 promised "exact source offsets", which reads as
raw-file offsets.

Resolution, [`1cf8b71`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/1cf8b71):

- New `rag_engine.documents`: `documents.jsonl` holds the full `Document`
  (doc_id, corpus_id, source, content_hash, normalized text, sections,
  metadata). On read, `Document` re-validates `doc_id` and `content_hash`.
- `rag-engine ingest` writes `<stem>.documents.jsonl` next to the chunks,
  verifying them and writing the documents first.
- `rag-engine index` requires that file for a chunks input.
- Every index directory stores `documents.jsonl`.
- `provenance()` gains `content_hash`.
- README now says offsets index the normalized document text, not raw-file
  bytes.
- [ADR-0006](docs/adr/0006-corpus-scoped-ids-and-verifiable-offsets.md)
  records the decision.

### MAJOR 2: `doc_id = sha256(relative path)` depends on the ingest root and collides across corpora

Resolution, [`a18aca9`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/a18aca9):

- `doc_id = sha256("<corpus_id>:<source>")[:16]`.
- `corpus_id` must match `^[a-z0-9][a-z0-9._-]{0,63}$`. It cannot contain
  `:`, so the key is unambiguous.
- `source` must be a normalized relative POSIX path.
- `corpus_id` defaults to the slugified corpus root directory name. Set it
  with `--corpus-id`, `RAG_CORPUS_ID`, or `IngestionConfig.corpus_id`.
- The scheme is documented in `docs/ingestion-and-chunking.md` and ADR-0006.
- `tests/test_document_ids.py` covers:
  - the pinned id `make_doc_id("handbook", "guide/intro.md") == "120aa960849299ee"`;
  - the same tree under two different roots with one corpus id giving equal
    doc and chunk ids;
  - two corpora with the same relative path never colliding.
- The index format change is noted in
  [`1cf8b71`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/1cf8b71).
  The manifest moves to `format_version` 2, and format-1 indexes are rejected
  with a request to rebuild.

### MINOR findings

| Finding | Resolution |
| --- | --- |
| `Chunk` does not validate `end_char > start_char` (`models.py:112-113`) | `Chunk` now requires `end_char > start_char` and `len(text) == end_char - start_char` ([`a18aca9`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/a18aca9)). |
| Offsets are not re-checked after load | `load_index` runs `verify_chunks` on both sub-indexes against `documents.jsonl`, and `save_index` and `index` do the same before writing ([`1cf8b71`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/1cf8b71)). |
| No end-to-end offset-invariant test | `tests/test_offsets_end_to_end.py` and `tests/test_offsets_fusion_rerank.py` ([`eeca981`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/eeca981)) run the loaders (Markdown with front matter and CRLF, ligatures, HTML, text with control and full-width characters) through all three chunkers, save/load, RRF, weighted fusion, rerank and nested rerank. At every stage they check `document.text[start:end] == chunk.text` and the `content_hash`. |
| Provenance lacks `page_end` | `provenance()` includes `page_end`, which equals `page` unless the chunk spans pages ([`1cf8b71`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/1cf8b71)). |
| `fusion._resolve_weights` (`fusion.py:149`) accepts NaN/inf | Non-finite weights are rejected ([`b36d286`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/b36d286)). |
| Nested rerank overwrites `base_rank` and the base score (`rerank.py:161-164`) | `RetrievedChunk.ranks` is a new field, keyed by stage name like `components`. Each rerank stage adds its base stage's score and rank under that stage's own name, and refuses to overwrite an existing key. Ranks are no longer stored as floats among the scores ([`b36d286`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/b36d286)). |

### OPTIONAL

| Finding | Resolution |
| --- | --- |
| `schema_version` on the search JSON output | `search --json` prints `{"schema_version": 1, "query", "mode", "rerank", "hits"}`, documented in the README. This is a breaking change from the earlier bare list ([`8bf4367`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/8bf4367)). |

### Also fixed while resolving the review

Found by the automated code review of each fix:

- Front-matter keys can no longer shadow `source`, `corpus_id` or
  `content_hash` in chunk metadata (`a18aca9`).
- A rejected `save_index` (for example, duplicate documents) fails before
  touching an existing index (`1cf8b71`).

### Sign-off: Chad's re-check at `cdd892e` (2026-10-09)

**Passed for the v0.1.0 interface release.** Both MAJORs and all MINORs are
closed. The re-check confirmed three things:

- citations are verified against the stored `content_hash`;
- document ids are stable and scoped by corpus;
- reranking keeps each stage's rank.

The final review gate still runs after M7.

Non-blocking advice: always pass `--corpus-id` (or `RAG_CORPUS_ID`), because
the default falls back to the folder name. The README's usage section now says
so and uses `--corpus-id` in its examples.

## Chad's review of M4 at `96b3c33` (2026-10-10)

Scope: the M4 evaluation harness. Result: the metrics math is correct, with
1 MAJOR and 4 MINOR findings. All of them are resolved.

### MAJOR: modes were not compared at equal depth (`runner.py:301`)

Dense and lexical fetched `top_k` (10) chunks and hybrid fetched `candidates`
(50). Hits were then collapsed to unique documents and cut to 10, so a baseline
could be scored on fewer than 10 documents. On real SciFact, lexical was short
on 9/300 queries.

Resolution, [`9f236ba`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/9f236ba):

- Every mode starts at the same chunk depth: `EvalConfig.depth`, default 50,
  raised to at least `top_k`.
- `retrieve_documents` doubles the depth until it has `top_k` unique documents,
  or until the index or the optional `max_depth` cap is exhausted.
- Each report records per-mode `initial_depth`, `max_depth`,
  `max_depth_used`, `mean_depth_used` and `queries_short_of_k`.
- `rag-engine evaluate` gains `--depth` and `--max-depth`.
- `tests/test_runner_depth.py` builds a corpus where one document owns every
  top chunk. It checks that all modes still reach `top_k` unique documents by
  over-fetching, and that a tight `max_depth` cap is reported as short.

### MINOR findings

| Finding | Resolution |
| --- | --- |
| No warm-up before latency timing | Each mode runs one untimed warm-up retrieval before timing. It uses a fixed query that is never evaluated, so no scored query is pre-cached ([`9f236ba`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/9f236ba)). |
| Report records the git SHA but not a dirty working tree | Reports carry `git_dirty` next to `git_sha` (from `git status --porcelain --untracked-files=no`; `null` outside a checkout) ([`9f236ba`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/9f236ba)). |
| Queries whose qrels are all non-relevant are scored 0 | They are skipped and counted in `n_queries_skipped_no_relevant`. Queries with no qrels are counted in `n_queries_unlabelled`. `n_queries` counts scored queries only ([`9f236ba`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/9f236ba)). |
| Structured set (3 docs / 8 queries) presented like a benchmark | Chosen fix: label it smoke-only rather than score by section. Its reports set `smoke_only: true` with a note, and the README and `evals/README.md` say so. Section-level scoring via `relevant_sections` was not adopted: it would add a second metric definition for an 8-query set that cannot support quality claims anyway ([`9f236ba`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/9f236ba)). |

### Also fixed

- Coverage had dropped to 95.55% at `96b3c33`, below the 96% bar.
  [`e718f74`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/e718f74) adds error-path tests for the structured loader, the
  runner and `evaluate --dataset scifact`.
- The automated review of the fix led to two more changes: the warm-up query
  no longer coincides with the first scored query, and `queries_short_of_k`
  is documented as covering both index exhaustion and the depth cap.
- The report `schema_version` is now 2. The committed hashing smoke reports
  were regenerated at `e718f74` with the new schema and a clean tree.


## Chad's final review at `900575f` (2026-10-27)

Scope: the whole project before v1.0.0 (M1–M7). Result: 0 CRITICAL, 1 MAJOR,
2 MINOR. All three are resolved; Chad's re-check of the fixes is pending, and
v1.0.0 is not tagged until he signs off.

### MAJOR: the README has no latency numbers, and the spec's latency target is unchecked

PROJECT_SPEC §8 sets a target of p95 search latency under 100 ms without
reranking, and the README promised "per-stage latency", but no latency table
was published even though the committed SciFact report records it. The harness
times each query end to end per mode, not per stage. The README results table
also showed only the @10 metrics, though the reports have @1 and @5.

Resolution, [`aa1427b`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/aa1427b)
and [`ff1619c`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/ff1619c):

- `scripts/eval_table.py --report <file>` renders one report: nDCG@10, MRR@10,
  Recall@{1,5,10} and Hit@{1,5,10} per mode, a p50/p95 retrieval latency table
  per mode (`n/a` where a mode has no timing), and the report's hardware record
  verbatim. `tests/test_eval_table.py` covers it, including a run on the
  committed report.
- The README's SciFact quality table now has the @1 and @5 columns, from that
  output. Both fastembed reports hold identical metrics.
- New README section "SciFact retrieval latency per mode", from
  `scifact_test_fastembed_20261012T013527Z.json` (`99f28f6`, `git_dirty:
  false`, 300 queries). Its p95 values are dense 40.1 ms, lexical 7.7 ms, hybrid
  61.4 ms and hybrid+rerank 1751.6 ms. The hardware is stated as the report
  records it: platform `Linux-6.12.94+-x86_64-with-glibc2.41`, processor
  empty, python `3.13.5`, cpu_count `8`. The section also says that the report
  has no CPU model and that the run was not on a CI runner.
- The §8 target is **met** on that machine: the slowest mode without reranking
  is hybrid at a p95 of 61.4 ms. §8 no longer says "on a CI-class CPU", because
  CI runs no models. It now names the corpus and points to the harness
  measurement on the machine the report records.
- "Per-stage latency" now reads "latency per mode" in the README and in the
  spec's goals. The README and spec §7 say what one timing covers and that stage
  timings and index build time are not recorded.

### MINOR findings

| Finding | Resolution |
| --- | --- |
| PROJECT_SPEC §7 still said answer quality would be "not run" until a provider was configured | §7 now describes the recorded M5 run: 30 seeded SciFact claims (20 answerable, 10 with gold removed), `cline-pass/deepseek-v4.1-flash` at temperature 0, `c55a2bb`, graded by code only (parse rate, span validity with the 20-character minimum, citation recall/precision vs qrels gold, false and correct abstention), Wilson intervals, and offline replay from the cache ([`ff1619c`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/ff1619c)). |
| The README does not list the fenced-code limitation of heading stripping | New "Known limitations" section in the README. Its first entry: the extractive generator scans each chunk on its own, so a chunk that starts inside a fenced code block takes `#` comment lines in it for headings, reads the closing fence as an opening one, and can then quote a real heading after it. Checked on the box before writing it up. The section also lists one eval dataset, the 30-claim answer eval, exact (linear) dense search and the unauthenticated read-only service ([`ff1619c`](https://github.com/miguelzzzzzzzz/CitationNeeded/commit/ff1619c)). |

### For the re-check

- The README latency table and the @1/@5 columns should match
  `python scripts/eval_table.py --report evals/results/scifact_test_fastembed_20261012T013527Z.json`.
- Is the §8 rewording (machine recorded in the report, not a CI-class CPU)
  acceptable, or should the target stay as written and be marked unverified?
