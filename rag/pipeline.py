from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

from config import AppConfig
from .chunking import TextChunker
from .embeddings import SentenceEmbeddingEncoder
from .generator import LocalChatGenerator
from .loaders import DocumentLoader
from .prompts import build_prompt
from .rerank import CrossEncoderReranker
from .schemas import AnswerResult, IndexReport
from .store import InMemoryVectorStore


LOGGER = logging.getLogger(__name__)


class ModularRAGPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.loader = DocumentLoader(config)
        self.chunker = TextChunker(config)
        self.store = InMemoryVectorStore()

        self._embedder: SentenceEmbeddingEncoder | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._generator: LocalChatGenerator | None = None

        self._chunk_count = 0
        self._document_count = 0
        self._sources: list[str] = []

    def build_index(self, file_paths: list[str | Path]) -> IndexReport:
        start = perf_counter()
        documents = self.loader.load_many(file_paths)
        LOGGER.info("文档加载完成，共 %s 个文档单元。", len(documents))
        chunks = self.chunker.split_documents(documents)
        if not chunks:
            raise ValueError("没有切分出可用文本块，请检查上传文档内容。")
        LOGGER.info("切块完成，共 %s 个文本块。", len(chunks))

        embedder = self._get_embedder()
        embeddings = embedder.encode_corpus([chunk.text for chunk in chunks])
        self.store.build(chunks, embeddings)

        self._document_count = len(documents)
        self._chunk_count = len(chunks)
        self._sources = sorted({doc.source_name for doc in documents})
        elapsed = perf_counter() - start
        LOGGER.info("索引构建完成，耗时 %.2f 秒。", elapsed)

        return IndexReport(
            file_count=len(file_paths),
            document_count=self._document_count,
            chunk_count=self._chunk_count,
            sources=self._sources,
        )

    def answer(self, question: str, retrieve_top_k: int | None = None, rerank_top_k: int | None = None) -> AnswerResult:
        if not question.strip():
            raise ValueError("问题不能为空。")
        if not self.store.is_ready:
            raise RuntimeError("知识库尚未构建，请先上传文档并点击“建立索引”。")

        retrieve_top_k = retrieve_top_k or self.config.retrieve_top_k
        rerank_top_k = rerank_top_k or self.config.rerank_top_k
        LOGGER.info("开始执行问答链路 retrieve_top_k=%s rerank_top_k=%s", retrieve_top_k, rerank_top_k)

        embedder = self._get_embedder()
        reranker = self._get_reranker()
        generator = self._get_generator()

        query_embedding = embedder.encode_query(question)
        retrieved = self.store.search(query_embedding, top_k=retrieve_top_k)
        LOGGER.info("向量召回完成，候选数=%s", len(retrieved))
        reranked = reranker.rerank(question, retrieved, top_k=rerank_top_k)
        LOGGER.info("重排完成，返回数=%s", len(reranked))
        prompt = build_prompt(question, reranked)
        answer = generator.generate(prompt)
        LOGGER.info("生成完成，回答长度=%s 字符。", len(answer))
        return AnswerResult(answer=answer, retrieved=reranked, prompt=prompt)

    def clear(self) -> None:
        self.store.clear()
        self._chunk_count = 0
        self._document_count = 0
        self._sources = []

    @property
    def is_ready(self) -> bool:
        return self.store.is_ready

    @property
    def source_names(self) -> list[str]:
        return list(self._sources)

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    def _get_embedder(self) -> SentenceEmbeddingEncoder:
        if self._embedder is None:
            self._embedder = SentenceEmbeddingEncoder(self.config)
        return self._embedder

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.config)
        return self._reranker

    def _get_generator(self) -> LocalChatGenerator:
        if self._generator is None:
            self._generator = LocalChatGenerator(self.config)
        return self._generator
