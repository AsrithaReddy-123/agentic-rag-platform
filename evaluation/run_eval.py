"""Run the retrieval and answer benchmark and write evaluation/results/benchmark.json."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from app.config import Settings
from app.engine import build_engine
from app.guardrails import groundedness
from app.logging_config import log_event
from evaluation.dataset import CORPUS_VERSION, build_knowledge_base, split_questions
from evaluation.metrics import mean, percentile, precision_at_k, recall_at_k, reciprocal_rank, token_f1
from evaluation.ragas_eval import run_ragas

RETRIEVAL_MODES = (
    ("vector_only", "vector"),
    ("bm25_only", "bm25"),
    ("hybrid_bm25_vector", "hybrid"),
    ("hybrid_cross_encoder", "hybrid_rerank"),
)


def _round(value: float) -> float:
    return round(float(value), 4)


def retrieval_report(questions, search, mode: str) -> dict:
    recalls: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []
    latencies: list[float] = []
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"recall": [], "mrr": []})
    for question in questions:
        if question.qtype == "unanswerable":
            continue
        started = time.perf_counter()
        hits = search(question.text, k=5, mode=mode, filters=question.filters)
        latencies.append((time.perf_counter() - started) * 1000)
        ids = [hit["id"] for hit in hits]
        recall = recall_at_k(ids, question.gold_ids, 5)
        precision = precision_at_k(ids, question.gold_ids, 5)
        rank = reciprocal_rank(ids, question.gold_ids)
        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(rank)
        by_type[question.qtype]["recall"].append(recall)
        by_type[question.qtype]["mrr"].append(rank)
    return {
        "recall@5": _round(mean(recalls)),
        "precision@5": _round(mean(precisions)),
        "mrr": _round(mean(mrrs)),
        "p95_retrieval_ms": _round(percentile(latencies, 95)),
        "mean_retrieval_ms": _round(mean(latencies)),
        "questions": len(recalls),
        "by_type": {
            qtype: {"recall@5": _round(mean(bucket["recall"])), "mrr": _round(mean(bucket["mrr"]))}
            for qtype, bucket in sorted(by_type.items())
        },
    }


def answer_report(questions, answer_fn) -> dict:
    relevance: list[float] = []
    grounds: list[float] = []
    unsupported = 0
    abstained = 0
    latencies: list[float] = []
    for question in questions:
        started = time.perf_counter()
        result = answer_fn(question.text, filters=question.filters)
        latencies.append((time.perf_counter() - started) * 1000)
        evidence = " ".join(hit["text"] for hit in result.get("sources", []))
        if result["abstained"]:
            abstained += 1
            score = 1.0
        else:
            score = groundedness(result["answer"], evidence)
            if score < 0.85:
                unsupported += 1
        grounds.append(score)
        if question.qtype != "unanswerable":
            relevance.append(0.0 if result["abstained"] else token_f1(result["answer"], question.answer))
    total = len(questions) or 1
    return {
        "relevance_f1": _round(mean(relevance)),
        "groundedness": _round(mean(grounds)),
        "unsupported_rate": _round(unsupported / total),
        "abstain_rate": _round(abstained / total),
        "p95_e2e_ms": _round(percentile(latencies, 95)),
        "mean_e2e_ms": _round(mean(latencies)),
        "questions": len(questions),
    }


def select_threshold(pairs: list[tuple[float, bool]]) -> dict:
    confidences = [confidence for confidence, _flag in pairs]
    candidates = [min(confidences) - 1.0, *sorted(set(confidences)), max(confidences) + 1.0]
    best: tuple[tuple, float, float, float] | None = None
    for threshold in candidates:
        unanswerable = [confidence for confidence, flag in pairs if flag]
        answerable = [confidence for confidence, flag in pairs if not flag]
        unanswerable_abstain = sum(confidence < threshold for confidence in unanswerable) / len(unanswerable)
        answerable_kept = sum(confidence >= threshold for confidence in answerable) / len(answerable)
        feasible = unanswerable_abstain >= 0.75
        key = (feasible, unanswerable_abstain + answerable_kept)
        if best is None or key > best[0]:
            best = (key, threshold, unanswerable_abstain, answerable_kept)
    assert best is not None
    return {
        "abstain_threshold": _round(best[1]),
        "dev_unanswerable_abstain_rate": _round(best[2]),
        "dev_answerable_keep_rate": _round(best[3]),
        "dev_questions": len(pairs),
    }


def log_mlflow(payload: dict) -> str:
    try:
        import mlflow
    except ImportError:
        return "mlflow_not_installed"
    try:
        import os

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment("agentic-rag")
        with mlflow.start_run(run_name="benchmark"):
            flat: dict[str, float] = {}
            for system, metrics in payload["retrieval"].items():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        flat[f"retrieval.{system}.{key}"] = float(value)
            for system, metrics in payload["answers"].items():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        flat[f"answers.{system}.{key}"] = float(value)
            mlflow.log_metrics(flat)
        return "logged"
    except Exception as exc:  # noqa: BLE001 - benchmark should still write JSON if MLflow fails
        return f"skipped:{type(exc).__name__}"


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Benchmark results",
        "",
        "Generated by `python -m evaluation.run_eval`. Do not edit by hand.",
        "",
        f"- Corpus chunks: {payload['corpus_chunks']}",
        f"- Held-out questions: {payload['evaluation_questions']}",
        f"- Embedder: `{payload['embedder']}`",
        f"- Reranker: `{payload['reranker']}`",
        "",
        "## Retrieval",
        "",
        "| System | Recall@5 | Precision@5 | MRR | P95 retrieval |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in payload["retrieval"].items():
        if "recall@5" not in metrics:
            continue
        lines.append(
            f"| {name} | {metrics['recall@5']:.3f} | {metrics['precision@5']:.3f} | "
            f"{metrics['mrr']:.3f} | {metrics['p95_retrieval_ms']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "| System | Relevance F1 | Groundedness | Unsupported rate | Abstain rate | P95 e2e |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in payload["answers"].items():
        lines.append(
            f"| {name} | {metrics['relevance_f1']:.3f} | {metrics['groundedness']:.3f} | "
            f"{metrics['unsupported_rate']:.3f} | {metrics['abstain_rate']:.3f} | {metrics['p95_e2e_ms']:.1f} ms |"
        )
    rag = payload["answers"]["rag_unvalidated"]["unsupported_rate"]
    agentic = payload["answers"]["agentic_rag"]["unsupported_rate"]
    if rag > 0:
        reduction = (rag - agentic) / rag
        lines.extend(["", f"Unsupported-rate reduction, unvalidated RAG to agentic RAG: {reduction:.1%}.", ""])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--output", default="evaluation/results/benchmark.json")
    args = parser.parse_args()

    chunks, questions = build_knowledge_base(args.chunks, seed=args.seed)
    splits = split_questions(questions, seed=args.seed)
    settings = Settings(embedder=args.embedder, reranker=args.reranker, corpus_size=len(chunks), abstain_threshold=-1e9)
    engine = build_engine(settings, chunks=chunks)
    for question in splits["test"][:3]:
        engine.search(question.text, k=5, mode="hybrid_rerank")

    retrieval = {}
    for label, mode in RETRIEVAL_MODES:
        log_event("eval_retrieval", mode=label)
        retrieval[label] = retrieval_report(splits["test"], engine.search, mode)

    dev_pairs = []
    for question in splits["dev"]:
        result = engine.answer(question.text, mode="agentic", filters=question.filters, abstain_threshold=-1e9)
        dev_pairs.append((result["confidence"], question.qtype == "unanswerable"))
    threshold = select_threshold(dev_pairs)

    def answerable_only(mode: str):
        def _run(question: str, filters: dict | None = None) -> dict:
            return engine.answer(question, mode=mode, filters=filters, abstain_threshold=threshold["abstain_threshold"])

        return _run

    answers = {
        "llm_only": answer_report(splits["test"], answerable_only("llm_only")),
        "rag_unvalidated": answer_report(splits["test"], answerable_only("rag_unvalidated")),
        "agentic_rag": answer_report(splits["test"], answerable_only("agentic")),
    }
    agentic_hits = []

    class _Capture:
        def __call__(self, question: str, k: int = 5, mode: str = "hybrid_rerank", filters: dict | None = None):
            del k, mode
            result = engine.answer(
                question,
                mode="agentic",
                filters=filters,
                abstain_threshold=threshold["abstain_threshold"],
            )
            return result["retrieved"]

    retrieval["agentic_graph"] = retrieval_report(splits["test"], _Capture(), "agentic")
    del agentic_hits

    ragas_samples = []
    for question in splits["test"]:
        if question.qtype == "unanswerable":
            continue
        result = engine.answer(
            question.text,
            mode="agentic",
            filters=question.filters,
            abstain_threshold=threshold["abstain_threshold"],
        )
        if result["abstained"]:
            continue
        ragas_samples.append(
            {
                "question": question.text,
                "answer": result["answer"],
                "contexts": [hit["text"] for hit in result["sources"]],
                "ground_truth": question.answer,
            }
        )
        if len(ragas_samples) >= 30:
            break

    payload = {
        "project": "agentic-rag-platform",
        "corpus_version": CORPUS_VERSION,
        "corpus_chunks": len(chunks),
        "evaluation_questions": sum(1 for q in splits["test"] if q.qtype != "unanswerable"),
        "abstention_questions": sum(1 for q in splits["test"] if q.qtype == "unanswerable"),
        "embedder": engine.embedder.model_name,
        "reranker": engine.reranker.model_name,
        "seed": args.seed,
        "threshold": threshold,
        "retrieval": retrieval,
        "answers": answers,
        "ragas": run_ragas(ragas_samples),
        "notes": (
            "Synthetic curated policies. Retrieval metrics use held-out questions with known chunk ids. "
            "The abstention threshold is selected on a disjoint development split. "
            "Groundedness is lexical overlap between the answer and cited evidence."
        ),
    }
    payload["mlflow"] = log_mlflow(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(output.with_name("BENCHMARK.md"), payload)
    print(output.read_text())


if __name__ == "__main__":
    main()
