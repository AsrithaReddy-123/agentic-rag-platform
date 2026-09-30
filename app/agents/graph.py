"""Multi-agent graph: decompose, retrieve, validate, generate, abstain."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.generate import compose_answer
from app.guardrails import groundedness, is_supported
from app.retrieval.hybrid import HybridRetriever


class RAGState(TypedDict, total=False):
    question: str
    filters: dict
    subqueries: list[str]
    retrieved: list[dict]
    evidence: list[dict]
    answer: str
    citations: list[str]
    confidence: float
    abstained: bool
    trace: list[str]
    abstain_threshold: float
    support_threshold: float


def _decompose(question: str) -> list[str]:
    marker = " Additionally, "
    if marker in question:
        parts = [part.strip() for part in question.split(marker) if part.strip()]
        if len(parts) >= 2:
            return parts
    return [question]


def build_graph(searcher, embedder):
    def decompose(state: RAGState) -> dict:
        subqueries = _decompose(state["question"])
        trace = list(state.get("trace", []))
        trace.append(f"decompose:{len(subqueries)}")
        return {"subqueries": subqueries, "trace": trace}

    def retrieve(state: RAGState) -> dict:
        retriever = HybridRetriever(
            searcher=searcher,
            mode="hybrid_rerank",
            k=5,
            filters=state.get("filters") or {},
        )
        hits: list[dict] = []
        seen: set[str] = set()
        for subquery in state["subqueries"]:
            for doc in retriever.invoke(subquery):
                doc_id = str(doc.metadata["id"])
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                metadata = {key: str(value) for key, value in doc.metadata.items() if key not in {"id", "score"}}
                hits.append(
                    {
                        "id": doc_id,
                        "text": doc.page_content,
                        "score": float(doc.metadata.get("score", 0.0)),
                        "metadata": metadata,
                    }
                )
        trace = list(state.get("trace", []))
        trace.append(f"retrieve:{len(hits)}")
        return {"retrieved": hits, "trace": trace}

    def validate(state: RAGState) -> dict:
        evidence = list(state.get("retrieved", []))[:4]
        trace = list(state.get("trace", []))
        trace.append(f"validate:{len(evidence)}")
        return {"evidence": evidence, "trace": trace}

    def route(state: RAGState) -> str:
        evidence = state.get("evidence") or []
        if not evidence:
            return "abstain"
        threshold = float(state.get("abstain_threshold", 0.0))
        if float(evidence[0]["score"]) < threshold:
            return "abstain"
        return "generate"

    def generate(state: RAGState) -> dict:
        answer, citations, confidence = compose_answer(
            state["question"],
            state.get("evidence") or [],
            embedder,
            hallucinate=False,
        )
        evidence_text = " ".join(item["text"] for item in state.get("evidence") or [])
        support_threshold = float(state.get("support_threshold", 0.85))
        if not answer or not is_supported(answer, evidence_text, support_threshold):
            trace = list(state.get("trace", []))
            trace.append("generate:unsupported")
            return {
                "answer": "I don't have enough supported evidence in the knowledge base to answer that.",
                "citations": [],
                "confidence": confidence,
                "abstained": True,
                "trace": trace,
            }
        trace = list(state.get("trace", []))
        trace.append(f"generate:citations={len(citations)}:groundedness={groundedness(answer, evidence_text):.2f}")
        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "abstained": False,
            "trace": trace,
        }

    def abstain(state: RAGState) -> dict:
        trace = list(state.get("trace", []))
        trace.append("abstain")
        confidence = float(state["evidence"][0]["score"]) if state.get("evidence") else 0.0
        return {
            "answer": "I don't have enough supported evidence in the knowledge base to answer that.",
            "citations": [],
            "confidence": confidence,
            "abstained": True,
            "trace": trace,
        }

    graph = StateGraph(RAGState)
    graph.add_node("decompose", decompose)
    graph.add_node("retrieve", retrieve)
    graph.add_node("validate", validate)
    graph.add_node("generate", generate)
    graph.add_node("abstain", abstain)
    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "validate")
    graph.add_conditional_edges("validate", route, {"generate": "generate", "abstain": "abstain"})
    graph.add_edge("generate", END)
    graph.add_edge("abstain", END)
    return graph.compile()
