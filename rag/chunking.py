from __future__ import annotations

import re
from typing import Iterable

from config import AppConfig
from .schemas import DocumentChunk, SourceDocument


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？；!?;\n])")
_WHITESPACE_PATTERN = re.compile(r"[ \t\u3000]+")


class TextChunker:
    def __init__(self, config: AppConfig):
        self.config = config

    def split_documents(self, documents: Iterable[SourceDocument]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for document in documents:
            text_chunks = self._split_text(document.text)
            for chunk_index, chunk_text in enumerate(text_chunks, start=1):
                metadata = dict(document.metadata)
                metadata["chunk_index"] = chunk_index
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document.doc_id}-chunk-{chunk_index}",
                        source_name=document.source_name,
                        text=chunk_text,
                        metadata=metadata,
                    )
                )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        pieces = [piece.strip() for piece in _SENTENCE_SPLIT_PATTERN.split(normalized) if piece.strip()]
        if not pieces:
            return []

        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if len(piece) > self.config.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._slide(piece))
                continue

            proposed = piece if not current else current + piece
            if len(proposed) <= self.config.chunk_size:
                current = proposed
            else:
                chunks.append(current)
                overlap = current[-self.config.chunk_overlap :] if self.config.chunk_overlap > 0 else ""
                current = overlap + piece
                if len(current) > self.config.chunk_size:
                    chunks.extend(self._slide(current))
                    current = ""

        if current:
            chunks.append(current)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _slide(self, text: str) -> list[str]:
        step = max(1, self.config.chunk_size - self.config.chunk_overlap)
        windows: list[str] = []
        for start in range(0, len(text), step):
            window = text[start : start + self.config.chunk_size].strip()
            if window:
                windows.append(window)
            if start + self.config.chunk_size >= len(text):
                break
        return windows

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [_WHITESPACE_PATTERN.sub(" ", line).strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)
