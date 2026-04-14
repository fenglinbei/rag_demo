from __future__ import annotations

import hashlib
from pathlib import Path

import fitz  # PyMuPDF

from config import AppConfig
from .schemas import SourceDocument


class DocumentLoader:
    def __init__(self, config: AppConfig):
        self.config = config

    def load_many(self, file_paths: list[str | Path]) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        for raw_path in file_paths:
            path = Path(raw_path)
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                documents.extend(self._load_pdf(path))
            elif suffix in self.config.supported_text_suffixes:
                documents.append(self._load_text(path))
            else:
                raise ValueError(
                    f"暂不支持文件类型：{path.suffix or '[无扩展名]'}。"
                    f"当前仅支持 PDF 和常见纯文本文件。"
                )
        return documents

    def _load_text(self, path: Path) -> SourceDocument:
        text = self._read_text(path)
        text = text.strip()
        if not text:
            raise ValueError(f"文本文件为空：{path.name}")
        return SourceDocument(
            doc_id=self._stable_id(path.name),
            source_name=path.name,
            text=text,
            metadata={"type": "text", "filename": path.name},
        )

    def _load_pdf(self, path: Path) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        with fitz.open(path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                page_text = page.get_text("text", sort=True).strip()
                if not page_text:
                    continue
                documents.append(
                    SourceDocument(
                        doc_id=self._stable_id(f"{path.name}-page-{page_index}"),
                        source_name=path.name,
                        text=page_text,
                        metadata={
                            "type": "pdf",
                            "filename": path.name,
                            "page": page_index,
                        },
                    )
                )

        if not documents:
            raise ValueError(
                f"PDF {path.name} 没有提取到可用文本。若它是扫描版 PDF，本 demo 暂不包含 OCR。"
            )
        return documents

    def _read_text(self, path: Path) -> str:
        for encoding in self.config.encoding_candidates:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法读取文本文件编码：{path.name}")

    @staticmethod
    def _stable_id(raw_text: str) -> str:
        return hashlib.md5(raw_text.encode("utf-8")).hexdigest()
