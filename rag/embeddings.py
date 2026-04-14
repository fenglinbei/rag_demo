from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from config import AppConfig
from .model_utils import ensure_model_path


LOGGER = logging.getLogger(__name__)


class SentenceEmbeddingEncoder:
    def __init__(self, config: AppConfig):
        self.config = config
        LOGGER.info("加载 embedding 模型: %s", config.embedding_model_name)
        ensure_model_path(config.embedding_model_name, "Embedding")
        self.model = SentenceTransformer(
            config.embedding_model_name,
            device=config.device,
            cache_folder=str(config.cache_dir),
        )

    def encode_corpus(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_query(self, query: str) -> np.ndarray:
        query_text = f"{self.config.query_instruction}{query}" if self.config.query_instruction else query
        return self.model.encode(
            [query_text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
