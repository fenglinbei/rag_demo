from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent


@dataclass(slots=True)
class AppConfig:
    """Global configuration for the demo project.

    Most fields can be overridden with environment variables so the project can
    be adapted without changing the code.
    """

    project_name: str = "本地中文 RAG Demo"
    cache_dir: Path = BASE_DIR / ".cache"
    supported_text_suffixes: tuple[str, ...] = (
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".log",
        ".rst",
    )

    embedding_model_name: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "./models/bge-small-zh-v1.5"
    )
    reranker_model_name: str = os.getenv(
        "RAG_RERANKER_MODEL", "./models/bge-reranker-base"
    )
    generator_model_name: str = os.getenv(
        "RAG_LLM_MODEL", "./models/Qwen2.5-3B-Instruct"
    )

    device: str = os.getenv(
        "RAG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_4bit: bool = os.getenv("RAG_USE_4BIT", "1") == "1"

    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "450"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
    retrieve_top_k: int = int(os.getenv("RAG_RETRIEVE_TOP_K", "8"))
    rerank_top_k: int = int(os.getenv("RAG_RERANK_TOP_K", "3"))
    reranker_max_length: int = int(os.getenv("RAG_RERANK_MAX_LENGTH", "512"))

    max_new_tokens: int = int(os.getenv("RAG_MAX_NEW_TOKENS", "256"))
    temperature: float = float(os.getenv("RAG_TEMPERATURE", "0.2"))
    repetition_penalty: float = float(os.getenv("RAG_REPETITION_PENALTY", "1.05"))

    query_instruction: str = os.getenv(
        "RAG_QUERY_INSTRUCTION", "为这个问题生成表示以检索相关中文资料："
    )
    log_level: str = os.getenv("RAG_LOG_LEVEL", "INFO")
    log_file: Path = Path(os.getenv("RAG_LOG_FILE", str(BASE_DIR / "rag_demo.log")))
    encoding_candidates: tuple[str, ...] = field(
        default_factory=lambda: ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1")
    )

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)


CONFIG = AppConfig()
CONFIG.ensure_dirs()
