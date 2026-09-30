"""Reciprocal rank fusion of dense retrieval and BM25, with optional reranking."""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.retrieval.bm25 import BM25Index
from app.schemas import Chunk


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridSearcher:
    def __init__(self, chunks: list[Chunk], dense_store, reranker) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.ordered = list(chunks)
        self.dense = dense_store
        self.bm25 = BM25Index(chunks)
        self.reranker = reranker

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid_rerank",
        filters: dict[str, str] | None = None,
    ) -> list[dict]:
        filters = filters or {}
        if mode == "vector":
            return self.dense.search(query, k=k, filters=filters)
        if mode == "bm25":
            return self.bm25.search(query, k=k, filters=filters)
        if mode not in {"hybrid", "hybrid_rerank"}:
            raise ValueError(f"unknown retrieval mode: {mode}")
        dense_hits = self.dense.search(query, k=40, filters=filters)
        sparse_hits = self.bm25.search(query, k=40, filters=filters)
        fused = reciprocal_rank_fusion(
            [[hit["id"] for hit in dense_hits], [hit["id"] for hit in sparse_hits]]
        )
        if mode == "hybrid_rerank":
            head = [doc_id for doc_id, _score in fused[:20]]
            reranked = self.reranker.rerank(query, head, self.chunks)
            return [self._hit(doc_id, score) for doc_id, score in reranked[:k]]
        return [self._hit(doc_id, score) for doc_id, score in fused[:k]]

    def _hit(self, doc_id: str, score: float) -> dict:
        chunk = self.chunks[doc_id]
        return {"id": chunk.id, "text": chunk.text, "score": float(score), "metadata": chunk.metadata}

    def add(self, chunk: Chunk) -> None:
        self.ordered.append(chunk)
        self.chunks[chunk.id] = chunk
        self.bm25 = BM25Index(self.ordered)
        self.dense.add(chunk)


class HybridRetriever(BaseRetriever):
    """LangChain retriever used by the agent graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    searcher: HybridSearcher
    mode: str = "hybrid_rerank"
    k: int = 5
    filters: dict = {}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        hits = self.searcher.search(query, k=self.k, mode=self.mode, filters=self.filters)
        documents = []
        for hit in hits:
            metadata = {"id": hit["id"], "score": hit["score"], **hit["metadata"]}
            documents.append(Document(page_content=hit["text"], metadata=metadata))
        return documents
