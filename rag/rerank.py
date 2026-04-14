from __future__ import annotations

from sentence_transformers import CrossEncoder

from config import AppConfig
from .schemas import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = CrossEncoder(
            config.reranker_model_name,
            max_length=config.reranker_max_length,
            device=config.device,
            trust_remote_code=True,
            automodel_args={"cache_dir": str(config.cache_dir)},
        )

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)

        rescored: list[RetrievedChunk] = []
        for candidate, score in zip(candidates, scores, strict=True):
            rescored.append(
                RetrievedChunk(
                    chunk=candidate.chunk,
                    retrieval_score=candidate.retrieval_score,
                    rerank_score=float(score),
                )
            )

        rescored.sort(key=lambda item: item.rerank_score if item.rerank_score is not None else -1e9, reverse=True)
        return rescored[: max(1, min(top_k, len(rescored)))]
