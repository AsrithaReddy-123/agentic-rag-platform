from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_and_query_shape():
    app = create_app(Settings(embedder="hash", reranker="overlap", corpus_size=400, abstain_threshold=-1e9))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        ready = client.get("/ready").json()
        assert ready["status"] == "ready"
        assert ready["chunks"] == 400
        response = client.post("/query", json={"question": "What does policy vac-de-40 require?"})
        assert response.status_code == 200
        body = response.json()
        assert "answer" in body
        assert "citations" in body
        assert "abstained" in body
        stats = client.get("/stats").json()
        assert stats["queries"] == 1
