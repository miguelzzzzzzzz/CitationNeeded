# production-rag-engine

A production-style retrieval-augmented generation (RAG) engine. Work in progress; see `PLAN.md`.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest
```
