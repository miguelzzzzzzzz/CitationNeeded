# Ingestion and chunking

## Pipeline

```
path -> ingest_path() -> [Document] -> build_chunker(config).chunk(doc) -> [Chunk] -> JSONL
```

### Loaders

| Format | Extensions | Structure extracted | Metadata extracted |
| --- | --- | --- | --- |
| Markdown | `.md .markdown .mdx` | ATX headings (ignored inside fenced code) -> nested heading paths | YAML front matter (title, tags, ...), heading count |
| HTML | `.html .htm .xhtml` | `h1`-`h6` -> heading paths; lists, tables, `pre` kept as text | `<title>`, `lang`, meta description/author/keywords |
| PDF | `.pdf` | one section per page (`page` set) | PDF title/author, page count, empty pages |
| Text | `.txt .text .rst .log` | single section | encoding |

All loaders record file name, extension, size, and word count. Text is
normalized (NFKC, line endings, control characters, whitespace runs, optional
PDF de-hyphenation) before sectioning, so section and chunk offsets refer to
the normalized text that is stored.

Failure handling is per file. `IngestionReport.skipped` lists unsupported
extensions, files over `RAG_MAX_FILE_BYTES`, binary content, files with no
extractable text (including scanned PDFs, as OCR is out of scope), unreadable
PDFs, and exact-duplicate content. Hidden files and directories are ignored and
symlinks are not followed unless configured.

Document ids are derived from the source path relative to the ingestion root,
so re-ingesting an edited file keeps its id while `content_hash` changes; this
is what incremental re-indexing (M2) will key on.

### Chunkers

| Strategy | How it splits | When to use |
| --- | --- | --- |
| `fixed` | sliding window of `chunk_size` tokens, `chunk_overlap` overlap | baseline; uniform sizes |
| `recursive` | paragraph -> line -> sentence -> word boundaries, then greedy packing with overlap of whole pieces | unstructured prose |
| `structure` | `recursive` applied inside each section (heading or page) | Markdown/HTML/PDF with meaningful structure (default) |

Shared behavior:

- **Exact provenance:** every chunk satisfies
  `document.text[chunk.start_char:chunk.end_char] == chunk.text`.
- **Size bound:** no chunk exceeds `chunk_size` whitespace tokens (ADR-0005).
- **Small-chunk merging:** chunks below `min_chunk_tokens` are merged into a
  neighbour when the result still fits; the merged chunk keeps only the heading
  path common to both parts.
- **Context header:** `Chunk.embedding_text` prefixes
  `Title > Heading > Subheading` (configurable) so short sections remain
  retrievable by their headings.
- **Deterministic ids:** derived from document id, content hash, and span.

These properties are tested over every strategy x several size/overlap
settings x randomly generated multi-section documents, plus the fixture corpus
(`tests/test_chunking.py`).

## Comparing strategies

```bash
python scripts/compare_chunking.py tests/fixtures/corpus --sizes 16 32 64
```

Output on the three fixture documents (descriptive statistics, not a quality
benchmark; run at the commit that added this document):

| strategy | size | overlap | chunks | mean tokens | p95 tokens | crossing sections |
| --- | --- | --- | --- | --- | --- | --- |
| fixed | 16 | 2 | 11 | 13.73 | 16.0 | 45% |
| recursive | 16 | 2 | 11 | 12.64 | 15.5 | 27% |
| structure | 16 | 2 | 11 | 12.45 | 16.0 | 0% |
| fixed | 32 | 4 | 6 | 24.5 | 32.0 | 67% |
| recursive | 32 | 4 | 6 | 23.33 | 28.5 | 50% |
| structure | 32 | 4 | 6 | 22.5 | 28.5 | 33% |
| fixed | 64 | 8 | 4 | 35.75 | 59.5 | 75% |
| recursive | 64 | 8 | 4 | 35.0 | 51.85 | 75% |
| structure | 64 | 8 | 4 | 33.75 | 52.7 | 75% |

Observation: the fixture sections are very short (roughly 10-20 tokens), so
with the default `min_chunk_tokens=16` the structure-aware chunker merges
neighbouring sections and its boundary-crossing rate converges to the other
strategies as `chunk_size` grows. Whether section-pure chunks actually improve
retrieval is an empirical question for the M4 structured-document eval set;
no retrieval-quality claim is made here.

## Known limitations

- Token budgets are whitespace tokens, not model tokens (ADR-0005).
- HTML `<pre>` indentation is not preserved; tables are flattened to `|`-separated text.
- PDF extraction depends on the PDF's text layer (no OCR, no layout analysis).
- Setext (underlined) Markdown headings are not recognized; ATX (`#`) headings are.
