---
title: Retrieval Engineering Guide
tags: [rag, retrieval]
version: 2
---

Intro paragraph before the first heading.

# Retrieval Engineering Guide

Dense retrieval maps queries and passages into a shared vector space.

## Lexical search

BM25 scores documents by term frequency, inverse document frequency, and
document length normalization.

```python
# this comment is not a heading
score = bm25(query, doc)
```

### Tuning k1 and b

The k1 parameter controls term-frequency saturation; b controls length normalization.

## Hybrid search

Reciprocal rank fusion combines ranked lists without calibrating scores.
