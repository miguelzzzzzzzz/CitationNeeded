# ADR-0003: SciFact (BEIR) as the primary retrieval evaluation corpus

Status: Accepted (2026-10-09)

## Context
Retrieval claims must come from executed evaluations on labeled data with a
clear license, small enough to embed on a CPU in minutes.

## Decision
Use SciFact as distributed by BEIR: 5,183 abstracts, 300 test queries with
relevance judgments. Licenses: claims and annotations CC BY 4.0; abstracts
ODC-By 1.0 (S2ORC). Data is downloaded by a script with a checksum and is
never committed. Results are reported alongside published BEIR baselines only
as context, never as our own numbers.

## Alternatives considered
- **MS MARCO**: non-commercial terms and far too large for CPU-only runs.
- **NFCorpus / FiQA**: viable; may be added as secondary datasets.
- **Synthetic LLM-generated queries**: useful later, but needs an LLM and
  risks evaluating the generator rather than the retriever.

## Consequences
SciFact documents are single abstracts, so it does not exercise
structure-aware chunking. A small hand-labeled, multi-section Markdown set
will be added in M4 for chunking experiments.
