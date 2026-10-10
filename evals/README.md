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
