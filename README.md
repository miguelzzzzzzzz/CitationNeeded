# Citation Needed

A production-style hybrid RAG service that answers with citations to exact source spans.

[![CI](https://github.com/miguelzzzzzzzz/CitationNeeded/actions/workflows/ci.yml/badge.svg)](https://github.com/miguelzzzzzzzz/CitationNeeded/actions/workflows/ci.yml)

A retrieval-augmented generation engine built the way a production team would
build one: typed, swappable pipeline stages; structure-aware chunking with
exact source offsets; hybrid dense + BM25 retrieval with reranking; cited,
schema-validated answers; and a reproducible evaluation harness so every
design choice is backed by measured retrieval quality and latency.

> **Status:** early development (see [PLAN.md](PLAN.md)). M1 (ingestion and
> chunking) and M2 (embeddings, vector index, BM25, retrievers, index/search CLI)
> are done; M3 (hybrid fusion and reranking) is next.
> No benchmark results exist yet; none will be shown here until produced by a
> committed evaluation run.

## Why this is not "chat with a PDF"

| Typical demo | This project |
| --- | --- |
| One loader, fixed-size chunks | Markdown/HTML/text/PDF loaders, three chunking strategies, heading-path and page metadata |
| Top-k cosine only | Dense + BM25 hybrid with rank fusion, metadata filters, cross-encoder reranking |
| Free-text answer | Pydantic-validated response with citations to exact character spans |
| "It looks right" | Recall@K, MRR, Hit Rate, nDCG and per-stage latency on SciFact (BEIR) |
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
rag-engine ingest path/to/docs --out chunks.jsonl --strategy structure --chunk-size 256 --chunk-overlap 32
```

This loads Markdown, HTML, PDF, and text files, writes one JSON chunk per line
(text, exact character span, heading path, page, metadata), and prints chunk
statistics plus any skipped files with the reason. Details:
[docs/ingestion-and-chunking.md](docs/ingestion-and-chunking.md).

Build an index directory (dense vectors + BM25) and query it:

```bash
rag-engine index path/to/docs --out index/            # or a chunks.jsonl from `ingest`
rag-engine search index/ "how does reranking work" -k 5            # dense (bge-small)
rag-engine search index/ "BM25 k1" --mode lexical --filter source=guide.md --json
```

Each hit carries its rank, score, retriever, and provenance (document and chunk
id, source file, heading path, page, character span). `--embedder hashing`
builds a fully offline index for testing. Combining dense and lexical results
(hybrid fusion) arrives in M3.

## Naming

The repository is called *Citation Needed*. The Python distribution
(`production-rag-engine`), import package (`rag_engine`), and CLI
(`rag-engine`) keep their descriptive names.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
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
