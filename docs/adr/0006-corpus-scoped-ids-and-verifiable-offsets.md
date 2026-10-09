# ADR-0006: Corpus-scoped document ids and verifiable offsets

- Status: Accepted
- Date: 2026-10-09
- Context: review of `84bce05` (see [REVIEW.md](../../REVIEW.md))

## Context

Document ids were `sha256(relative path)[:16]`. The relative path depends on
which directory was passed to `ingest`, so the same file got a different id when
ingested from a parent directory, and two corpora with the same layout (for
example two `README.md` files) shared ids. Chunk offsets pointed into normalized
text (NFKC, LF newlines, control characters removed) that was not persisted, so
an index could not show that a citation's span really says what the chunk says.

## Decision

1. `doc_id = sha256("<corpus_id>:<source>")[:16]`. `source` must be a normalized
   relative POSIX path. `corpus_id` matches `^[a-z0-9][a-z0-9._-]{0,63}$`, so it
   cannot contain the `:` separator. It defaults to the slugified corpus root
   directory name and is configurable (`--corpus-id`, `RAG_CORPUS_ID`,
   `IngestionConfig.corpus_id`).
2. The normalized documents are persisted: `ingest` writes
   `<stem>.documents.jsonl` beside the chunks file, and index directories hold
   `documents.jsonl`. `Document` validates `doc_id` and `content_hash` on load,
   and `verify_chunks` checks that every chunk's document exists, its recorded
   `content_hash` and `source` match, and `chunk.text ==
   document.text[start_char:end_char]`.
3. The index format version is now 2; format-1 indexes are rejected with a
   request to rebuild rather than migrated.

## Consequences

- Citations can be re-checked offline from the index alone.
- The default corpus id is only as stable as the directory name; two corpora in
  directories with the same name need explicit ids.
- Indexes and chunk files from before this change must be rebuilt.
- `documents.jsonl` roughly doubles on-disk text; acceptable for the corpus
  sizes targeted (see ADR-0001).
