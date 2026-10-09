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

