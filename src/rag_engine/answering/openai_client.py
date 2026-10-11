"""OpenAI-compatible chat-completions answer generator (ADR-0004).

Deliberately stdlib-only: any OpenAI-compatible ``/chat/completions`` endpoint is
reachable through ``urllib.request``, so the optional LLM path adds no third-party
dependency. Configuration is injected by the caller (which reads ``RAG_LLM_*``),
so this module keeps no global state and is trivial to point at a fake server.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from rag_engine.answering.models import AnswerResponse, citation_from_provenance
from rag_engine.retrieval.retriever import RetrievedChunk

__all__ = ["OpenAICompatibleAnswerGenerator"]

_SCHEMA = (
    '{"answer": <string>, "citations": [{"chunk_index": <int>, "quote": <string>}], '
    '"abstained": <bool>}'
)

# Models often wrap JSON in a markdown fence, sometimes with a prose preamble.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_SYSTEM_PROMPT = (
    "You answer questions strictly from the numbered contexts supplied by the user. "
    "Reply with exactly one JSON object and nothing else: no prose, no markdown fences. "
    f"The object must match {_SCHEMA}. "
    "chunk_index is the 0-based index of the context block it cites, i.e. the number in "
    "'[0]', '[1]', ... . Every quote must be copied verbatim from that context's text. "
    "When the contexts do not contain the answer, reply with "
    '{"answer": "", "citations": [], "abstained": true} instead of guessing.'
)

_REPAIR_PROMPT = (
    "Your previous reply was rejected: {error}. "
    "Reply again with ONLY one JSON object matching {schema}, with no prose and no "
    "markdown fences."
)


class OpenAICompatibleAnswerGenerator:
    """Answer generator backed by an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_repair_attempts: int = 1,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not model.strip():
            raise ValueError("model must be a non-empty string")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than 0")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if temperature < 0:
            raise ValueError("temperature must not be negative")
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must not be negative")

        self._base_url = base_url.rstrip("/")
        self._model = model
        # Normalise a blank key to None so the Authorization header is never empty.
        self._api_key = api_key.strip() if api_key else None
        self._timeout_s = float(timeout_s)
        self._max_tokens = max_tokens
        self._temperature = float(temperature)
        self._max_repair_attempts = max_repair_attempts

    @property
    def name(self) -> str:
        return "openai"

    def generate(
        self,
        query: str,
        hits: Sequence[RetrievedChunk],
        *,
        max_sentences: int = 5,
    ) -> AnswerResponse:
        """Answer ``query`` from ``hits``, repairing malformed model JSON once."""
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        if max_sentences < 1:
            raise ValueError("max_sentences must be at least 1")

        contexts = list(hits)
        if not contexts:
            return self._abstain(query)

        messages = self._build_messages(query, contexts, max_sentences=max_sentences)
        content = self._chat(messages)
        repairs_left = self._max_repair_attempts

        while True:
            try:
                payload = self._parse_content(content)
                return self._to_answer(query, payload, contexts)
            except ValueError as exc:
                if repairs_left <= 0:
                    raise ValueError(
                        f"model output still unusable after repair attempts: {exc}"
                    ) from exc
                repairs_left -= 1
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": _REPAIR_PROMPT.format(error=exc, schema=_SCHEMA),
                    },
                ]
                content = self._chat(messages)

    # ---------------------------------------------------------------- transport

    def _chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        """POST one chat completion and return ``choices[0].message.content``."""
        body = json.dumps(
            {
                "model": self._model,
                "messages": [dict(message) for message in messages],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw: bytes = response.read()
        except urllib.error.HTTPError as exc:
            raise ValueError(f"LLM endpoint returned HTTP {exc.code} ({exc.reason})") from exc
        except urllib.error.URLError as exc:
            raise ValueError(f"LLM endpoint unreachable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ValueError(f"LLM request timed out after {self._timeout_s:g}s") from exc
        return self._extract_content(raw)

    @staticmethod
    def _extract_content(raw: bytes) -> str:
        try:
            envelope = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("LLM endpoint returned a non-JSON body") from exc
        if not isinstance(envelope, dict):
            raise ValueError("LLM response body is not a JSON object")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM response contains no choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("LLM response choice has no textual message content")
        return content

    # ------------------------------------------------------------------ parsing

    @staticmethod
    def _strip_fences(content: str) -> str:
        match = _FENCE_RE.search(content)
        return match.group(1).strip() if match is not None else content.strip()

    @classmethod
    def _parse_content(cls, content: str) -> dict[str, Any]:
        text = cls._strip_fences(content)
        if not text:
            raise ValueError("model returned empty content")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"model output is not valid JSON ({exc.msg})") from exc
        if not isinstance(payload, dict):
            raise ValueError("model output must be a JSON object")
        return payload

    def _to_answer(
        self,
        query: str,
        payload: Mapping[str, Any],
        contexts: Sequence[RetrievedChunk],
    ) -> AnswerResponse:
        abstained = payload.get("abstained", False)
        if not isinstance(abstained, bool):
            raise ValueError("'abstained' must be a boolean")
        raw_answer = payload.get("answer", "")
        if not isinstance(raw_answer, str):
            raise ValueError("'answer' must be a string")
        answer = raw_answer.strip()
        if abstained or not answer:
            return self._abstain(query)

        raw_citations = payload.get("citations", [])
        if not isinstance(raw_citations, list):
            raise ValueError("'citations' must be a list")
        citations = self._build_citations(raw_citations, contexts)
        return AnswerResponse(
            query=query,
            answer=answer,
            citations=citations,
            abstained=False,
            generator="openai",
            model_name=self._model,
        )

    @classmethod
    def _build_citations(
        cls,
        raw_citations: Sequence[Any],
        contexts: Sequence[RetrievedChunk],
    ) -> tuple[Any, ...]:
        """Map model citations onto provenance, dropping any that cannot be grounded."""
        built: list[Any] = []
        for entry in raw_citations:
            if not isinstance(entry, dict):
                continue
            index = entry.get("chunk_index")
            quote = entry.get("quote")
            if not isinstance(index, int) or isinstance(index, bool):
                continue
            if not isinstance(quote, str) or index < 0 or index >= len(contexts):
                continue
            chunk = contexts[index].chunk
            span = cls._locate(quote, chunk.text)
            if span is None:
                continue
            local_start, local_end = span
            quoted = chunk.text[local_start:local_end]
            if not quoted.strip():
                continue
            meta = chunk.metadata
            source = meta.get("source")
            content_hash = meta.get("content_hash")
            page_end = meta.get("page_end", chunk.page)
            built.append(
                citation_from_provenance(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    quote=quoted,
                    source=source if isinstance(source, str) else None,
                    content_hash=content_hash if isinstance(content_hash, str) else None,
                    heading_path=chunk.heading_path,
                    page=chunk.page,
                    page_end=page_end if isinstance(page_end, int) else chunk.page,
                    start_char=chunk.start_char + local_start,
                    end_char=chunk.start_char + local_end,
                )
            )
        return tuple(built)

    @staticmethod
    def _locate(quote: str, text: str) -> tuple[int, int] | None:
        """Return local offsets of ``quote`` in ``text``, or None if it is absent.

        Casefold offsets are an approximation: casefolding may change a character's
        length (e.g. ``ß`` -> ``ss``), so the recovered slice is still validated by
        the caller slicing ``text`` rather than the folded copy.
        """
        if not quote:
            return None
        start = text.find(quote)
        if start >= 0:
            return start, start + len(quote)
        # Case-insensitive fallback only when casefold preserves length (ASCII).
        folded_text = text.casefold()
        folded_quote = quote.casefold()
        if len(folded_text) != len(text) or len(folded_quote) != len(quote):
            return None
        start = folded_text.find(folded_quote)
        if start < 0:
            return None
        return start, start + len(quote)

    # ------------------------------------------------------------------ helpers

    def _build_messages(
        self,
        query: str,
        contexts: Sequence[RetrievedChunk],
        *,
        max_sentences: int,
    ) -> list[dict[str, str]]:
        blocks = []
        for index, hit in enumerate(contexts):
            chunk = hit.chunk
            source = chunk.metadata.get("source")
            blocks.append(
                f"[{index}] doc_id={chunk.doc_id} chunk_id={chunk.chunk_id} "
                f"source={source}\ntext={chunk.text}"
            )
        user = (
            f"Contexts ({len(blocks)}), each identified by its bracket index:\n\n"
            + "\n\n".join(blocks)
            + f"\n\nQuestion: {query.strip()}\n"
            f"Answer in at most {max_sentences} sentence(s) and reply with JSON only."
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _abstain(self, query: str) -> AnswerResponse:
        return AnswerResponse(
            query=query,
            answer="",
            citations=(),
            abstained=True,
            generator="openai",
            model_name=self._model,
        )
