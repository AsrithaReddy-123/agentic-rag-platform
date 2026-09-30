import os

import pytest

from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.vector_store import PgVectorStore
from app.schemas import Chunk


def test_pgvector_roundtrip():
    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        pytest.skip("DATABASE_URL is not set")
    store = PgVectorStore(dsn, HashingEmbedder())
    store.setup()
    store.upsert(
        [
            Chunk(
                id="policy-de",
                text="People employed in Germany may carry over unused vacation hours.",
                metadata={"department": "people", "policy_id": "vac-de-40", "doc_type": "policy", "anchor": "Germany", "alert": "", "kind": "vacation"},
            )
        ]
    )
    hits = store.search("Germany vacation hours", k=3, filters={"department": "people"})
    assert hits
    assert hits[0]["id"] == "policy-de"
