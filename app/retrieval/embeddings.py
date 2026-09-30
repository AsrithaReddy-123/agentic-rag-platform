"""Dense encoders. The benchmark uses MiniLM; tests use a stable hashing encoder."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from app.tokenize import tokenize


class HashingEmbedder:
    """Bag-of-tokens hashing encoder. Used by unit tests, not by the published benchmark."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.model_name = "hash"

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.md5(token.encode()).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, index] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=len(texts) >= 256,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def load_embedder(name: str) -> HashingEmbedder | SentenceTransformerEmbedder:
    if name == "hash":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(name)


def encode_corpus(embedder, texts: list[str], cache_path: Path | None) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            return cached
    matrix = embedder.encode(texts)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, matrix)
    return matrix


def cache_file(cache_dir: str | None, model_name: str, chunk_ids: list[str]) -> Path | None:
    if not cache_dir or model_name == "hash":
        return None
    digest = hashlib.sha256("\n".join(chunk_ids).encode()).hexdigest()[:16]
    safe_model = model_name.replace("/", "_")
    return Path(cache_dir) / f"{safe_model}-{digest}.npy"
