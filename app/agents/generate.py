"""Extractive answers copied from retrieved evidence, plus an unvalidated baseline."""

from __future__ import annotations

import numpy as np

from app.guardrails import UNSUPPORTED_CLAIM
from app.tokenize import split_sentences


def best_sentences(question: str, evidence: list[dict], embedder, limit: int = 2) -> list[tuple[str, str, float]]:
    sentences: list[tuple[str, str]] = []
    for item in evidence:
        for sentence in split_sentences(item["text"]):
            if len(sentence) < 40:
                continue
            sentences.append((sentence, item["id"]))
    if not sentences:
        return []
    vectors = embedder.encode([question, *[sentence for sentence, _doc_id in sentences]])
    query_vec = vectors[0]
    sims = vectors[1:] @ query_vec
    order = np.argsort(-sims)[:limit]
    return [(sentences[int(i)][0], sentences[int(i)][1], float(sims[int(i)])) for i in order]


def compose_answer(
    question: str,
    evidence: list[dict],
    embedder,
    *,
    hallucinate: bool = False,
) -> tuple[str, list[str], float]:
    chosen = best_sentences(question, evidence, embedder)
    if not chosen:
        return "", [], 0.0
    answer = " ".join(sentence for sentence, _doc_id, _score in chosen)
    citations: list[str] = []
    for _sentence, doc_id, _score in chosen:
        if doc_id not in citations:
            citations.append(doc_id)
    if hallucinate:
        answer = answer + UNSUPPORTED_CLAIM
    confidence = float(evidence[0]["score"]) if evidence else 0.0
    return answer, citations, confidence
