# Evaluations

Milestone M4 evaluation harness for Citation Needed.

## Layout

- `datasets/` (git-ignored): downloaded SciFact via BEIR
  (`rag_engine.eval.scifact.ensure_scifact`). Checksum
  `5f7d1de60b170fc8027bb7898e2efca1`. Never commit this data (ADR-0003).
- `structured/`: in-repo multi-section Markdown corpus with hand-labeled
  `qrels.jsonl` for structure-aware chunking experiments.
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

Reports land under `evals/results/`. Unit tests cover metrics against
hand-computed examples and loaders against tiny fixtures; they never download
SciFact.
