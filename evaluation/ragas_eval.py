"""Optional RAGAS judge. Classical metrics in run_eval.py do not need an API key."""

from __future__ import annotations

import os


def run_ragas(samples: list[dict]) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": (
                "OPENAI_API_KEY is not set. Recall@K, Precision@K, MRR, lexical groundedness, "
                "and answer relevance are still computed. Install the eval extra and set a key "
                "to add the RAGAS LLM judge."
            ),
        }
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        return {"status": "skipped", "reason": "ragas is not installed. Run: uv sync --extra eval"}
    if not samples:
        return {"status": "skipped", "reason": "no samples"}
    frame = Dataset.from_list(samples)
    result = evaluate(frame, metrics=[faithfulness, answer_relevancy])
    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    return {"status": "ok", "scores": {key: float(value) for key, value in scores.items()}}
