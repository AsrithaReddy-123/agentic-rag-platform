"""In-process counters exposed at /stats."""

from __future__ import annotations

from evaluation.metrics import mean, percentile


class Stats:
    def __init__(self) -> None:
        self.queries = 0
        self.abstentions = 0
        self.failures = 0
        self.latencies_ms: list[float] = []
        self.confidences: list[float] = []

    def record(self, latency_ms: float, abstained: bool, confidence: float) -> None:
        self.queries += 1
        self.latencies_ms.append(latency_ms)
        self.confidences.append(confidence)
        if abstained:
            self.abstentions += 1
        self.latencies_ms = self.latencies_ms[-500:]
        self.confidences = self.confidences[-500:]

    def snapshot(self) -> dict:
        return {
            "queries": self.queries,
            "abstentions": self.abstentions,
            "failures": self.failures,
            "abstain_rate": (self.abstentions / self.queries) if self.queries else 0.0,
            "p95_latency_ms": percentile(self.latencies_ms, 95),
            "mean_latency_ms": mean(self.latencies_ms),
            "mean_confidence": mean(self.confidences),
        }
