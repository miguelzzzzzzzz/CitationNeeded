# Evaluations

Milestone M4 evaluation harness for Citation Needed.

## Layout

- `datasets/` (git-ignored): downloaded SciFact via BEIR
  (`rag_engine.eval.scifact.ensure_scifact`). Checksum
  `5f7d1de60b170fc8027bb7898e2efca1`. Never commit this data (ADR-0003).
- `structured/`: in-repo multi-section Markdown corpus with hand-labeled
  `qrels.jsonl`. It is **smoke-only**: 3 documents and 8 queries, scored at
  document level (`relevant_sections` is not used for scoring). Its reports set
  `"smoke_only": true` and must not back quality claims.
- `results/`: committed JSON reports from real runs (config, dataset checksum,
  git SHA, hardware, metrics, latency). README numbers must come from these
  files only — generate a Markdown table with `python scripts/eval_table.py`.
  Each new run also writes a `*.per_query.json` sidecar (schema 1) for paired
  bootstrap comparisons (`scripts/bootstrap_ci.py`).

## Running

```bash
# Offline smoke on the structured set (hashing embedder; not a quality claim)
rag-engine evaluate --dataset structured --embedder hashing

# SciFact (downloads into evals/datasets/ on first run)
rag-engine evaluate --dataset scifact --embedder hashing
# Real dense quality needs: pip install '.[embeddings]' and --embedder fastembed
```

## Method (report schema 2)

- Every mode (dense, lexical, hybrid, hybrid+rerank) starts at the same chunk
  depth (`--depth`, default 50, at least `--top-k`). Each query doubles the
  depth until the hits collapse to `--top-k` unique documents, or until the
  index or `--max-depth` is exhausted. `retrieval[mode]` records
  `initial_depth`, `max_depth`, `max_depth_used`, `mean_depth_used` and
  `queries_short_of_k`.
- Each mode runs one untimed warm-up query (a fixed, non-evaluated text) before
  latency is timed. The timed latency includes any over-fetch rounds.
- Queries without qrels (`n_queries_unlabelled`), or whose qrels have no grade
  >= 1 (`n_queries_skipped_no_relevant`), are skipped rather than scored 0.
  `n_queries` counts scored queries.
- Reports record `git_sha` and `git_dirty`; only reports with
  `git_dirty: false` should be cited.

Reports land under `evals/results/`. Unit tests cover metrics against
hand-computed examples and loaders against tiny fixtures; they never download
SciFact.

## Committed reports

| File | Dataset | Embedder | smoke_only | Notes |
| --- | --- | --- | --- | --- |
| `results/structured_hashing_*.json` | structured | hashing | true | Offline smoke; not a quality claim |
| `results/scifact_test_hashing_*.json` | SciFact test | hashing | false | Harness smoke on SciFact; hashing embedder — not a quality claim |
| `results/scifact_test_fastembed_20261010T184148Z.json` | SciFact test | fastembed | false | Real dense/lexical/hybrid/hybrid+rerank run at equal depth 50 (`git_dirty: false`). Chad matched aggregates to 3 decimals (Oct 11). |
| `results/scifact_test_fastembed_20261012T013527Z.json` (+ `.per_query.json`, `.bootstrap_ci.json`) | SciFact test | fastembed | false | Clean re-run at `99f28f6`; aggregates identical to the 2026-10-10 report; sidecar means equal the aggregates; paired bootstrap CIs (seed 0, B=10,000). |

## Per-query sidecar (schema 1)

Each aggregate report `NAME.json` written by the runner also gets a sibling
`NAME.per_query.json` when `write_report(..., per_query=...)` is used (the
`rag-engine evaluate` path always does). The sidecar holds aligned per-query
metric dicts for every mode so paired comparisons are possible without
re-parsing rankings:

```json
{
  "schema_version": 1,
  "aggregate_report": "NAME.json",
  "created_at": "...",
  "git_sha": "...",
  "git_dirty": false,
  "dataset_name": "scifact/test",
  "query_ids": ["...", "..."],
  "modes": {
    "dense": { "<qid>": { "ndcg@10": 0.0, "mrr@10": 0.0, "...": 0.0 } },
    "hybrid": { "<qid>": { "...": 0.0 } }
  }
}
```

Paired bootstrap CI for hybrid vs dense (or any two modes):

```bash
python scripts/bootstrap_ci.py evals/results/NAME.per_query.json \
  --mode-a hybrid --mode-b dense --metric ndcg@10
```

The 2026-10-10 fastembed report predates the sidecar. The clean re-run
`scifact_test_fastembed_20261012T013527Z` has both files, and
`….bootstrap_ci.json` records the CLI output for hybrid−dense,
hybrid+rerank−hybrid, hybrid+rerank−dense, dense−lexical and hybrid−lexical on
nDCG@10 and Recall@10 (95%, B=10,000, seed 0), with `ci_excludes_zero` per
comparison. Only hybrid−lexical and dense−lexical exclude 0. Do not invent
per-query numbers.

## Answer eval (M5, code-graded)

`scripts/answer_eval.py` runs answer generation on a small fixed-seed SciFact
test subset and grades it **with code only** (no LLM judge):

- **Subset:** `pick_subsets` takes `--n-answerable` claims with
  `random.Random(seed).sample(sorted(ids), n)` from the labelled test queries,
  then `--n-no-relevant` more from the rest with `seed + 1`.
- **Contexts:** hybrid retrieval (fastembed + BM25, RRF k=60), top `--k` (5)
  chunks. `no_relevant` items are a gold-removal ablation: the claim's qrels
  gold documents are removed from its retrieved list before taking `k`, so the
  correct behaviour is to abstain. SciFact qrels can be incomplete, so a few such
  items may still be answerable from non-gold abstracts.
- **Grades** (from the raw model JSON, because the generator silently drops
  ungrounded citations): citation span validity (quote occurs verbatim in the
  cited context), citation recall and precision against qrels gold docs present
  in the contexts, false abstention on answerable items with gold in context,
  correct abstention on `no_relevant` items, and a consistency check of the
  final response's document offsets. Unparseable replies count as failures.
- **Reproducibility:** every raw reply is stored in a JSONL cache under
  `evals/answers/`, keyed by sha256 of the exact request (endpoint, model,
  messages, temperature, max_tokens). `--replay` recomputes the metrics with no
  API call; a changed prompt, context or model is a cache miss, never a silent
  mismatch. Recording is done on a developer box, never in CI. CI tests use fake
  HTTP and pre-recorded caches only.
- **Provenance:** reports record the model id, `created_at`, `git_sha`,
  `git_dirty`, the selection method, the question template and the full config.
  Reports store SciFact claims (CC BY 4.0) and context ids plus text hashes, not
  abstract text; raw replies quote short abstract spans (ODC-By 1.0).

