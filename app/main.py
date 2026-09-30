"""FastAPI service for grounded question answering."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings
from app.engine import build_engine
from app.logging_config import configure_logging, log_event
from app.observability import Stats
from app.schemas import Chunk, IngestRequest, QueryRequest, QueryResponse


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        log_event("startup", embedder=settings.embedder, corpus_size=settings.corpus_size)
        app.state.engine = build_engine(settings)
        app.state.stats = Stats()
        log_event("index_ready", chunks=len(app.state.engine.chunks))
        yield

    app = FastAPI(title="Agentic RAG Platform", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        engine = getattr(app.state, "engine", None)
        if engine is None:
            return {"status": "starting"}
        return {"status": "ready", "chunks": len(engine.chunks)}

    @app.get("/stats")
    def stats() -> dict:
        return app.state.stats.snapshot()

    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        try:
            result = app.state.engine.answer(body.question, mode="agentic", filters=body.filters)
        except Exception:
            app.state.stats.failures += 1
            log_event("query_failed", question_chars=len(body.question))
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        app.state.stats.record(latency_ms, result["abstained"], result["confidence"])
        log_event(
            "query",
            latency_ms=round(latency_ms, 2),
            abstained=result["abstained"],
            citations=len(result["citations"]),
            question_chars=len(body.question),
        )
        sources = [
            {
                "id": hit["id"],
                "score": hit["score"],
                "policy_id": hit["metadata"].get("policy_id", ""),
                "snippet": hit["text"][:280],
            }
            for hit in result["sources"]
        ]
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            confidence=result["confidence"],
            abstained=result["abstained"],
            sources=sources,
            trace=result["trace"],
            latency_ms=latency_ms,
        )

    @app.post("/ingest")
    def ingest(body: IngestRequest) -> dict:
        app.state.engine.ingest(Chunk(id=body.id, text=body.text, metadata=body.metadata))
        log_event("ingest", chunk_id=body.id)
        return {"status": "indexed", "id": body.id, "chunks": len(app.state.engine.chunks)}

    return app


app = create_app()
