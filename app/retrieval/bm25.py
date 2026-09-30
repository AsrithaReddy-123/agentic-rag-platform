"""BM25 over the in-memory chunk corpus."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from app.schemas import Chunk
from app.tokenize import tokenize


def _passes(metadata: dict[str, str], filters: dict[str, str]) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        tokenized = [tokenize(chunk.text) for chunk in chunks]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5, filters: dict[str, str] | None = None) -> list[dict]:
        filters = filters or {}
        tokens = tokenize(query)
        if tokens == ["blank"]:
            return []
        scores = self._index.get_scores(tokens).astype(np.float64)
        for i, chunk in enumerate(self.chunks):
            if not _passes(chunk.metadata, filters):
                scores[i] = -1.0
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            # argpartition is enough, then sort the short head.
            pivot = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
            order = pivot[np.argsort(-scores[pivot])]
        hits = []
        for index in order:
            if scores[index] < 0:
                continue
            chunk = self.chunks[int(index)]
            hits.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "score": float(scores[index]),
                    "metadata": chunk.metadata,
                }
            )
            if len(hits) >= k:
                break
        return hits
