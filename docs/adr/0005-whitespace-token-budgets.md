# ADR-0005: Chunk budgets measured in whitespace tokens, with exact character offsets

Status: Accepted (2026-10-09)

## Context
Chunk sizes are usually expressed in model tokens, which ties chunking to one
tokenizer and adds a dependency. Citations need to point at exact source spans.

## Decision
Measure chunk size and overlap in whitespace-delimited tokens and store
`start_char`/`end_char` for every chunk, with the invariant
`document.text[start_char:end_char] == chunk.text`. A common rule of thumb for
English prose is ~1.3 subword tokens per word, so a 256-word chunk should fit
within a 512-token embedding window (bge-small); the actual ratio on the eval
corpus will be measured rather than assumed.

## Consequences
Deterministic, dependency-free chunking and precise citations. Budgets are
approximate with respect to a specific model's tokenizer; the embedder adapter
must truncate safely, and the ratio can be measured in M4.
