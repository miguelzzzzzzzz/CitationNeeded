"""Resolve a generator choice (``extractive``/``openai``/``auto``) to a generator.

Shared by ``rag-engine answer`` and ``POST /answer`` so both pick the backend
the same way: ``auto`` means the OpenAI-compatible client when
``RAG_LLM_BASE_URL`` and ``RAG_LLM_MODEL`` are both set, else the extractive
generator, which needs no model at all.
"""

from __future__ import annotations

from typing import Literal

from rag_engine.answering.base import AnswerGenerator
from rag_engine.answering.extractive import ExtractiveAnswerGenerator
from rag_engine.answering.openai_client import OpenAICompatibleAnswerGenerator
from rag_engine.config import LLMConfig

__all__ = ["GENERATOR_CHOICES", "GeneratorChoice", "select_generator"]

GeneratorChoice = Literal["extractive", "openai", "auto"]
GENERATOR_CHOICES: tuple[GeneratorChoice, ...] = ("extractive", "openai", "auto")


def select_generator(choice: str, llm: LLMConfig) -> AnswerGenerator:
    """Build the generator for ``choice``; ``ValueError`` if it cannot be served."""
    if choice not in GENERATOR_CHOICES:
        raise ValueError(f"unknown generator {choice!r}; expected one of {GENERATOR_CHOICES}")
    openai_ready = bool(llm.base_url and llm.model)
    if choice == "auto":
        choice = "openai" if openai_ready else "extractive"
    if choice == "extractive":
        return ExtractiveAnswerGenerator()
    if not openai_ready:
        raise ValueError(
            "openai generator requires RAG_LLM_BASE_URL and RAG_LLM_MODEL "
            "(or use the extractive generator)"
        )
    return OpenAICompatibleAnswerGenerator(
        base_url=llm.base_url or "",
        model=llm.model or "",
        api_key=llm.api_key,
        timeout_s=llm.timeout_s,
        max_tokens=llm.max_tokens,
        temperature=llm.temperature,
    )
