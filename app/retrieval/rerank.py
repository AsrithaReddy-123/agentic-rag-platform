"""Cross-encoder reranking. Tests can substitute a lexical overlap reranker."""

from __future__ import annotations

from app.tokenize import tokenize


def _token_f1(left: str, right: str) -> float:
    pred = set(tokenize(left))
    gold = set(tokenize(right))
    if not pred or not gold:
        return 0.0
    common = pred & gold
    if not common:
        return 0.0
    precision = len(common) / len(pred)
    recall = len(common) / len(gold)
    return 2 * precision * recall / (precision + recall)


class OverlapReranker:
    model_name = "overlap"

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [_token_f1(query, document) for query, document in pairs]

    def rerank(self, query: str, doc_ids: list[str], chunks: dict) -> list[tuple[str, float]]:
        pairs = [(query, chunks[doc_id].text) for doc_id in doc_ids]
        scores = self.score_pairs(pairs)
        ranked = sorted(zip(doc_ids, scores), key=lambda item: item[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in ranked]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        return [float(score) for score in scores]

    def rerank(self, query: str, doc_ids: list[str], chunks: dict) -> list[tuple[str, float]]:
        pairs = [(query, chunks[doc_id].text) for doc_id in doc_ids]
        scores = self.score_pairs(pairs)
        ranked = sorted(zip(doc_ids, scores), key=lambda item: item[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in ranked]


def load_reranker(name: str) -> OverlapReranker | CrossEncoderReranker:
    if name == "overlap":
        return OverlapReranker()
    return CrossEncoderReranker(name)
