"""Shared records for chunks, questions, and API payloads."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, str]


@dataclass
class Question:
    id: str
    text: str
    gold_ids: list[str]
    answer: str
    qtype: str
    department: str
    filters: dict[str, str] = field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str
    filters: dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    id: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    abstained: bool
    sources: list[dict]
    trace: list[str]
    latency_ms: float
