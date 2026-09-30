"""Wires the corpus, retrievers, and agent graph into one query engine."""

from __future__ import annotations

from app.agents.generate import compose_answer
from app.agents.graph import build_graph
from app.config import Settings
from app.retrieval.embeddings import load_embedder
from app.retrieval.hybrid import HybridSearcher
from app.retrieval.rerank import load_reranker
from app.retrieval.vector_store import NumpyVectorStore, PgVectorStore
from app.schemas import Chunk
from evaluation.dataset import build_knowledge_base


LLM_ONLY_ANSWER = (
    "Based on general knowledge, the standard company policy is to resolve the request "
    "within 5 business days by contacting the shared support inbox."
)


class Engine:
    def __init__(self, settings: Settings, chunks: list[Chunk] | None = None) -> None:
        self.settings = settings
        self.embedder = load_embedder(settings.embedder)
        self.reranker = load_reranker(settings.reranker)
        self.chunks = chunks if chunks is not None else build_knowledge_base(settings.corpus_size)[0]
        if settings.database_url:
            dense = PgVectorStore(settings.database_url, self.embedder)
            dense.setup()
            dense.upsert(self.chunks)
        else:
            dense = NumpyVectorStore(self.chunks, self.embedder, cache_dir=settings.cache_dir)
        self.searcher = HybridSearcher(self.chunks, dense, self.reranker)
        self.graph = build_graph(self.searcher, self.embedder)

    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank", filters: dict | None = None) -> list[dict]:
        return self.searcher.search(query, k=k, mode=mode, filters=filters or {})

    def answer(
        self,
        question: str,
        mode: str = "agentic",
        filters: dict | None = None,
        abstain_threshold: float | None = None,
    ) -> dict:
        filters = filters or {}
        threshold = self.settings.abstain_threshold if abstain_threshold is None else abstain_threshold
        if mode == "llm_only":
            return {
                "answer": LLM_ONLY_ANSWER,
                "citations": [],
                "confidence": 0.2,
                "abstained": False,
                "sources": [],
                "retrieved": [],
                "trace": ["llm_only"],
            }
        if mode == "rag_unvalidated":
            hits = self.search(question, k=3, mode="vector", filters=filters)
            answer, citations, confidence = compose_answer(question, hits, self.embedder, hallucinate=True)
            if not answer:
                answer = LLM_ONLY_ANSWER
            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "abstained": False,
                "sources": hits,
                "retrieved": hits,
                "trace": ["rag_unvalidated"],
            }
        state = self.graph.invoke(
            {
                "question": question,
                "filters": filters,
                "subqueries": [],
                "retrieved": [],
                "evidence": [],
                "answer": "",
                "citations": [],
                "confidence": 0.0,
                "abstained": False,
                "trace": [],
                "abstain_threshold": threshold,
                "support_threshold": self.settings.support_threshold,
            }
        )
        evidence = state.get("evidence") or []
        return {
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "confidence": float(state.get("confidence", 0.0)),
            "abstained": bool(state.get("abstained", False)),
            "sources": evidence,
            "retrieved": state.get("retrieved") or [],
            "trace": state.get("trace", []),
        }

    def ingest(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)
        self.searcher.add(chunk)


def build_engine(settings: Settings, chunks: list[Chunk] | None = None) -> Engine:
    return Engine(settings, chunks=chunks)
