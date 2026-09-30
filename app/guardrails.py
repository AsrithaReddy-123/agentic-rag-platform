"""Evidence support checks used before a response is returned."""

from __future__ import annotations

from app.tokenize import STOP, tokenize

UNSUPPORTED_CLAIM = (
    " The executive team has already waived this requirement for the current quarter."
)


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOP]


def groundedness(answer: str, evidence: str) -> float:
    """Share of answer content tokens that also appear in the cited evidence."""
    tokens = content_tokens(answer)
    if not tokens:
        return 0.0
    evidence_tokens = set(tokenize(evidence))
    return sum(token in evidence_tokens for token in tokens) / len(tokens)


def is_supported(answer: str, evidence: str, threshold: float = 0.85) -> bool:
    return groundedness(answer, evidence) >= threshold
