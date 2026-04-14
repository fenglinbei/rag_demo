from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceDocument:
    doc_id: str
    source_name: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    source_name: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: DocumentChunk
    retrieval_score: float
    rerank_score: float | None = None


@dataclass(slots=True)
class IndexReport:
    file_count: int
    document_count: int
    chunk_count: int
    sources: list[str]


@dataclass(slots=True)
class AnswerResult:
    answer: str
    retrieved: list[RetrievedChunk]
    prompt: str
