# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project specification, milestone plan, and ADRs 0001-0005 (vector index, local models, evaluation corpus, LLM provider interface, token budgets).
- Project scaffold: `pyproject.toml` (hatchling), MIT license, `.env.example`, `.gitignore`.
- Validated runtime settings (`rag_engine.config`) loaded from `RAG_*` environment variables.
- GitHub Actions CI: ruff lint + format check, strict mypy, pytest with coverage on Python 3.11 and 3.13.
