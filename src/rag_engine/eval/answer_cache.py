"""Committed, replayable cache of raw LLM replies for the answer-generation eval.

Recording happens once on a developer box (never in CI): a box-only script
calls the OpenAI-compatible endpoint and writes each raw reply into a JSONL
file. That file is committed to the repository so CI and everyone else can
replay the exact same responses offline. Cache keys hash the full request
(endpoint, model, messages, temperature, max_tokens, schema version), so any prompt,
context, or model change becomes a cache miss instead of a silent mismatch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from rag_engine.answering.models import AnswerResponse
from rag_engine.answering.openai_client import OpenAICompatibleAnswerGenerator
from rag_engine.retrieval.retriever import RetrievedChunk

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CacheMissError",
    "CachedAnswerGenerator",
    "ResponseCache",
    "cache_key",
]

CACHE_SCHEMA_VERSION: int = 1


def cache_key(
    *,
    base_url: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    payload = json.dumps(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "base_url": base_url.rstrip("/"),
            "model": model,
            "messages": [dict(m) for m in messages],
            "temperature": float(temperature),
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._entries: dict[str, str] = {}
        if self._path.exists():
            self._load()

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def get(self, key: str) -> str | None:
        return self._entries.get(key)

    def put(self, key: str, *, model: str, content: str) -> None:
        existing = self._entries.get(key)
        if existing is not None:
            if existing == content:
                return
            raise ValueError(f"{self._path}: key {key!r} already cached with different content")
        record = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": key,
            "model": model,
            "content": content,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._entries[key] = content

    def _load(self) -> None:
        text = self._path.read_text(encoding="utf-8")
        for lineno, raw in enumerate(text.splitlines(), start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self._path}: invalid JSON on line {lineno}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{self._path}: line {lineno} is not a JSON object")
            schema_version = record.get("schema_version")
            if schema_version != CACHE_SCHEMA_VERSION:
                raise ValueError(
                    f"{self._path}: line {lineno} has schema_version "
                    f"{schema_version!r}, expected {CACHE_SCHEMA_VERSION}"
                )
            key = record.get("key")
            model = record.get("model")
            content = record.get("content")
            if not isinstance(key, str):
                raise ValueError(f"{self._path}: line {lineno} has missing/non-str key")
            if not isinstance(model, str):
                raise ValueError(f"{self._path}: line {lineno} has missing/non-str model")
            if not isinstance(content, str):
                raise ValueError(f"{self._path}: line {lineno} has missing/non-str content")
            existing = self._entries.get(key)
            if existing is not None and existing != content:
                raise ValueError(
                    f"{self._path}: line {lineno} duplicates key {key!r} with different content"
                )
            self._entries[key] = content


class CacheMissError(LookupError):
    pass


class CachedAnswerGenerator(OpenAICompatibleAnswerGenerator):
    def __init__(
        self,
        *,
        cache: ResponseCache,
        replay_only: bool,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_repair_attempts: int = 1,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            temperature=temperature,
            max_repair_attempts=max_repair_attempts,
        )
        self._cache = cache
        self._replay_only = replay_only
        self.n_cache_hits = 0
        self.n_api_calls = 0
        self.raw_contents: list[str] = []

    @property
    def last_raw_content(self) -> str | None:
        if not self.raw_contents:
            return None
        return self.raw_contents[-1]

    def generate(
        self,
        query: str,
        hits: Sequence[RetrievedChunk],
        *,
        max_sentences: int = 5,
    ) -> AnswerResponse:
        self.raw_contents = []
        return super().generate(query, hits, max_sentences=max_sentences)

    def _chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        key = cache_key(
            base_url=self._base_url,
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        cached = self._cache.get(key)
        if cached is not None:
            self.n_cache_hits += 1
            content = cached
        elif self._replay_only:
            raise CacheMissError(
                f"no cached reply for key {key[:12]}… in {self._cache.path}; re-record with the API"
            )
        else:
            content = super()._chat(messages)
            self.n_api_calls += 1
            self._cache.put(key, model=self._model, content=content)
        self.raw_contents.append(content)
        return content
