"""Answer generation: extractive default and optional OpenAI-compatible client."""

from rag_engine.answering.base import AnswerGenerator
from rag_engine.answering.extractive import ExtractiveAnswerGenerator
from rag_engine.answering.models import (
    ANSWER_SCHEMA_VERSION,
    AnswerResponse,
    Citation,
    citation_from_provenance,
)
from rag_engine.answering.openai_client import OpenAICompatibleAnswerGenerator
from rag_engine.answering.select import GENERATOR_CHOICES, GeneratorChoice, select_generator

__all__ = [
    "ANSWER_SCHEMA_VERSION",
    "GENERATOR_CHOICES",
    "AnswerGenerator",
    "AnswerResponse",
    "Citation",
    "ExtractiveAnswerGenerator",
    "GeneratorChoice",
    "OpenAICompatibleAnswerGenerator",
    "citation_from_provenance",
    "select_generator",
]
