# Structured-document evaluation set

Small hand-labeled multi-section Markdown corpus for structure-aware chunking
experiments. SciFact abstracts are single-block documents and cannot exercise
heading-path provenance; these three docs can.

- `docs/`: three Markdown files with YAML front matter and H2 sections.
- `qrels.jsonl`: eight queries with relevant document stems and expected
  section heading paths.

Loaded by `rag_engine.eval.structured.load_structured_docs`. Not a substitute
for SciFact retrieval benchmarks; use it to compare chunking strategies.
