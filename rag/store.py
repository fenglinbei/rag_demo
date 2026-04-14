from __future__ import annotations

import numpy as np

from .schemas import DocumentChunk, RetrievedChunk


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._embeddings: np.ndarray | None = None

    @property
    def is_ready(self) -> bool:
        return self._embeddings is not None and len(self._chunks) > 0

    def build(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunk 数与 embedding 数不一致。")
        self._chunks = chunks
        self._embeddings = embeddings

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = None

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if not self.is_ready or self._embeddings is None:
            raise RuntimeError("索引尚未构建，请先上传文件并建立索引。")
        scores = self._embeddings @ query_embedding
        top_k = max(1, min(top_k, len(self._chunks)))
        top_indices = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(chunk=self._chunks[index], retrieval_score=float(scores[index]))
            for index in top_indices
        ]
