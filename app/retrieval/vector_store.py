"""Dense retrieval against a normalized numpy matrix or PostgreSQL pgvector."""

from __future__ import annotations

import numpy as np

from app.retrieval.embeddings import cache_file, encode_corpus
from app.schemas import Chunk


def _passes(metadata: dict[str, str], filters: dict[str, str]) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())


class NumpyVectorStore:
    def __init__(self, chunks: list[Chunk], embedder, cache_dir: str | None = None) -> None:
        self.chunks = chunks
        self.embedder = embedder
        path = cache_file(cache_dir, getattr(embedder, "model_name", "unknown"), [c.id for c in chunks])
        self.matrix = encode_corpus(embedder, [c.text for c in chunks], path)

    def search(self, query: str, k: int = 5, filters: dict[str, str] | None = None) -> list[dict]:
        filters = filters or {}
        query_vec = self.embedder.encode([query])[0]
        scores = self.matrix @ query_vec
        order = np.argsort(-scores)
        hits = []
        for index in order:
            chunk = self.chunks[int(index)]
            if not _passes(chunk.metadata, filters):
                continue
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

    def add(self, chunk: Chunk) -> None:
        vector = self.embedder.encode([chunk.text])
        self.chunks.append(chunk)
        self.matrix = np.vstack([self.matrix, vector])


class PgVectorStore:
    """PostgreSQL backend. The published benchmark uses NumpyVectorStore so it runs without a database."""

    def __init__(self, dsn: str, embedder) -> None:
        self.dsn = dsn
        self.embedder = embedder
        self.dim = embedder.dim

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self.dsn)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
        return conn

    def setup(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({self.dim})
                )
                """
            )
            conn.commit()

    def upsert(self, chunks: list[Chunk]) -> None:
        from psycopg.types.json import Json

        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        with self._connect() as conn:
            with conn.cursor() as cur:
                for chunk, vector in zip(chunks, vectors):
                    cur.execute(
                        """
                        INSERT INTO chunks (id, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        (chunk.id, chunk.text, Json(chunk.metadata), vector.tolist()),
                    )
            conn.commit()

    def search(self, query: str, k: int = 5, filters: dict[str, str] | None = None) -> list[dict]:
        filters = filters or {}
        query_vec = self.embedder.encode([query])[0].tolist()
        where = []
        params: list[object] = [query_vec]
        for key, value in filters.items():
            where.append("metadata->>%s = %s")
            params.extend([key, value])
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        params.extend([query_vec, k])
        sql = f"""
            SELECT id, text, metadata, 1 - (embedding <=> %s) AS score
            FROM chunks
            {where_sql}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {"id": row[0], "text": row[1], "score": float(row[3]), "metadata": dict(row[2])}
            for row in rows
        ]

    def add(self, chunk: Chunk) -> None:
        self.upsert([chunk])
