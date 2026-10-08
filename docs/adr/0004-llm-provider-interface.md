# ADR-0004: Provider-agnostic LLM interface with an extractive no-LLM default

Status: Accepted (2026-10-09)

## Context
Answer generation normally needs an LLM, but no paid key is authorized and the
system, tests, and evals must run with zero spend.

## Decision
Define an `AnswerGenerator` protocol. Provide (a) an extractive generator that
selects supporting sentences from top-ranked chunks and returns them with
citations, and (b) an OpenAI-compatible HTTP client configured by
`RAG_LLM_BASE_URL`, `RAG_LLM_MODEL`, `RAG_LLM_API_KEY`, which also works with
local servers (Ollama, vLLM, llama.cpp). Both return the same pydantic
response schema.

## Consequences
The whole pipeline runs and is tested offline. Generation-quality metrics
(faithfulness, citation precision) are scaffolded but reported as not run
until a provider is configured.
