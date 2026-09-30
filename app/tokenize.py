"""Tokenization shared by BM25, guardrails, and lexical metrics."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "is",
        "are",
        "be",
        "as",
        "at",
        "from",
        "that",
        "this",
        "it",
        "its",
        "into",
        "over",
        "above",
        "before",
        "after",
        "when",
        "what",
        "which",
        "who",
        "how",
        "does",
        "do",
        "within",
        "may",
        "must",
        "will",
        "their",
        "them",
        "they",
        "not",
        "than",
        "then",
        "per",
        "via",
        "can",
        "could",
        "about",
        "after",
        "also",
    }
)


def tokenize(text: str) -> list[str]:
    tokens = [tok for tok in _TOKEN.findall(text.lower()) if tok not in STOP and len(tok) > 1]
    return tokens or ["blank"]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]
