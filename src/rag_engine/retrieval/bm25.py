"""Okapi BM25 lexical index over chunks.

Scoring (Robertson/Sparck Jones with the non-negative IDF used by Lucene)::

    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(q, d) = sum over distinct terms t in q of
                  idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

``N`` is the number of indexed chunks, ``|d|`` the chunk's token count after
tokenization and ``avgdl`` the mean of those counts. Repeated query terms
count once. Only chunks containing at least one query term are returned.

The index is an inverted file (term -> {chunk_id: tf}) kept in memory, so
search touches only postings of the query terms. Like the vector store,
upserts replace whole documents, metadata filters are applied before
truncating to ``k``, and ties break by insertion order.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from rag_engine.models import Chunk
from rag_engine.retrieval.vector_store import Filters, SearchHit, matches

BM25_FORMAT_VERSION = 1
_WORD = re.compile(r"\w+")

Tokenizer = Callable[[str], list[str]]


def tokenize(text: str, stopwords: Iterable[str] = ()) -> list[str]:
    """NFKC-normalize, casefold, and split on Unicode word characters."""
    stop = frozenset(stopwords)
    tokens = _WORD.findall(unicodedata.normalize("NFKC", text).casefold())
    return [t for t in tokens if t not in stop] if stop else tokens


class BM25Index:
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Tokenizer = tokenize,
    ) -> None:
        if not (math.isfinite(k1) and k1 >= 0):
            raise ValueError("k1 must be a finite number >= 0")
        if not (0.0 <= b <= 1.0):
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self._tokenize = tokenizer
        self._chunks: dict[str, Chunk] = {}  # insertion-ordered
        self._order: dict[str, int] = {}
        self._lengths: dict[str, int] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._total_length = 0
        self._next = 0

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks.values())

    @property
    def average_length(self) -> float:
        return self._total_length / len(self._chunks) if self._chunks else 0.0

    def document_frequency(self, term: str) -> int:
        return len(self._postings.get(term, {}))

    def idf(self, term: str) -> float:
        n, df = len(self._chunks), self.document_frequency(term)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    # ------------------------------------------------------------------ mutation

    def _add(self, chunk: Chunk) -> None:
        counts = Counter(self._tokenize(chunk.embedding_text))
        self._chunks[chunk.chunk_id] = chunk
        self._order[chunk.chunk_id] = self._next
        self._next += 1
        length = sum(counts.values())
        self._lengths[chunk.chunk_id] = length
        self._total_length += length
        for term, tf in counts.items():
            self._postings.setdefault(term, {})[chunk.chunk_id] = tf

    def _remove(self, chunk_id: str) -> None:
        chunk = self._chunks.pop(chunk_id)
        del self._order[chunk_id]
        self._total_length -= self._lengths.pop(chunk_id)
        for term in set(self._tokenize(chunk.embedding_text)):
            postings = self._postings[term]
            del postings[chunk_id]
            if not postings:
                del self._postings[term]

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks; existing chunks of the same documents are replaced."""
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk_id in upsert batch")
        doc_ids = {c.doc_id for c in chunks}
        id_set = set(ids)
        stale = [cid for cid, c in self._chunks.items() if c.doc_id in doc_ids or cid in id_set]
        for chunk_id in stale:
            self._remove(chunk_id)
        for chunk in chunks:
            self._add(chunk)

    def delete_document(self, doc_id: str) -> int:
        stale = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        for chunk_id in stale:
            self._remove(chunk_id)
        return len(stale)

    # ------------------------------------------------------------------ search

    def scores(self, query: str) -> dict[str, float]:
        """BM25 score of every chunk that contains at least one query term."""
        if not self._chunks:
            return {}
        avgdl = self.average_length or 1.0
        totals: dict[str, float] = {}
        for term in dict.fromkeys(self._tokenize(query)):  # distinct, in order
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for chunk_id, tf in postings.items():
                norm = 1.0 - self.b + self.b * self._lengths[chunk_id] / avgdl
                gain = idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
                totals[chunk_id] = totals.get(chunk_id, 0.0) + gain
        return totals

    def search(self, query: str, k: int = 10, filters: Filters | None = None) -> list[SearchHit]:
        if k <= 0:
            raise ValueError("k must be positive")
        scored = self.scores(query)
        if filters:
            scored = {cid: s for cid, s in scored.items() if matches(self._chunks[cid], filters)}
        ranked = sorted(scored.items(), key=lambda item: (-item[1], self._order[item[0]]))[:k]
        return [
            SearchHit(chunk=self._chunks[cid], score=score, rank=rank)
            for rank, (cid, score) in enumerate(ranked, start=1)
        ]

    # ------------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> None:
        """Persist parameters and chunks; postings are rebuilt on load.

        A custom tokenizer is not serialized: load with the same tokenizer.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self._chunks.values():
                handle.write(chunk.model_dump_json() + "\n")
        manifest = {
            "format_version": BM25_FORMAT_VERSION,
            "k1": self.k1,
            "b": self.b,
            "count": len(self._chunks),
            "total_length": self._total_length,
        }
        (path / "bm25.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, tokenizer: Tokenizer = tokenize) -> BM25Index:
        path = Path(directory)
        manifest = json.loads((path / "bm25.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != BM25_FORMAT_VERSION:
            raise ValueError(f"unsupported BM25 format {manifest.get('format_version')!r}")
        index = cls(k1=float(manifest["k1"]), b=float(manifest["b"]), tokenizer=tokenizer)
        lines = (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        index.upsert([Chunk.model_validate_json(line) for line in lines if line.strip()])
        if len(index) != manifest["count"] or index._total_length != manifest["total_length"]:
            raise ValueError("BM25 files are inconsistent with manifest (tokenizer changed?)")
        return index
