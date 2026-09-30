"""Ranking and answer metrics. These are computed from system output, not hardcoded."""

from __future__ import annotations

import numpy as np

from app.tokenize import tokenize


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    relevant_ids = set(relevant)
    if not relevant_ids:
        raise ValueError("recall is undefined when there are no relevant ids")
    return len(set(retrieved[:k]) & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    relevant_ids = set(relevant)
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant_ids) / k


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    relevant_ids = set(relevant)
    for index, doc_id in enumerate(retrieved):
        if doc_id in relevant_ids:
            return 1.0 / (index + 1)
    return 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = set(tokenize(prediction))
    gold_tokens = set(tokenize(gold))
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = pred_tokens & gold_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))
