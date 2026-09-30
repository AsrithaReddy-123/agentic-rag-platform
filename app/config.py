"""Runtime settings. Environment variables override the defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    corpus_size: int = 2000
    database_url: str | None = None
    abstain_threshold: float = 0.0
    cache_dir: str = "data/cache"
    support_threshold: float = 0.85

    @classmethod
    def from_env(cls) -> Settings:
        database = os.getenv("DATABASE_URL", "").strip() or None
        return cls(
            embedder=os.getenv("EMBEDDER", cls.embedder),
            reranker=os.getenv("RERANKER", cls.reranker),
            corpus_size=int(os.getenv("CORPUS_SIZE", str(cls.corpus_size))),
            database_url=database,
            abstain_threshold=float(os.getenv("ABSTAIN_THRESHOLD", str(cls.abstain_threshold))),
            cache_dir=os.getenv("CACHE_DIR", cls.cache_dir),
            support_threshold=float(os.getenv("SUPPORT_THRESHOLD", str(cls.support_threshold))),
        )
