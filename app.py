from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from config import CONFIG
from rag import ModularRAGPipeline
from rag.logging_utils import setup_logging


setup_logging(CONFIG)
LOGGER = logging.getLogger(__name__)

PIPELINE = ModularRAGPipeline(CONFIG)
SUPPORTED_FILE_TYPES = [".pdf", *CONFIG.supported_text_suffixes]


def _format_build_report(report) -> str:
    source_preview = "\n".join(f"- {name}" for name in report.sources[:10])
    if len(report.sources) > 10:
        source_preview += f"\n- ... 共 {len(report.sources)} 个来源"
    return (
        f"已建立索引。\n\n"
        f"- 文件数：{report.file_count}\n"
        f"- 文档单元数：{report.document_count}\n"
        f"- 文本块数：{report.chunk_count}\n\n"
        f"已加载来源：\n{source_preview if source_preview else '- （空）'}"
    )


def build_index(uploaded_files: list[str] | None):
    if not uploaded_files:
        return "请先上传 PDF 或文本文件。", []

    try:
        LOGGER.info("开始建立索引，文件数=%s", len(uploaded_files))
        report = PIPELINE.build_index(uploaded_files)
        LOGGER.info(
            "索引建立完成，文档数=%s，文本块数=%s，来源数=%s",
            report.document_count,
            report.chunk_count,
            len(report.sources),
        )
        preview_rows = [[index + 1, name] for index, name in enumerate(report.sources)]
        return _format_build_report(report), preview_rows
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("建立索引失败: %s", exc)
        return f"建立索引失败：{exc}\n\n请查看日志文件：{CONFIG.log_file}", []


def answer_question(question: str, retrieve_top_k: int, rerank_top_k: int):
    if not PIPELINE.is_ready:
        return "请先上传文件并建立索引。", []

    try:
        LOGGER.info(
            "开始提问，问题长度=%s，retrieve_top_k=%s，rerank_top_k=%s",
            len(question.strip()),
            retrieve_top_k,
            rerank_top_k,
        )
        result = PIPELINE.answer(
            question=question,
            retrieve_top_k=int(retrieve_top_k),
            rerank_top_k=int(rerank_top_k),
        )
        LOGGER.info("提问完成，命中文本块=%s", len(result.retrieved))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("提问失败: %s", exc)
        return f"提问失败：{exc}\n\n请查看日志文件：{CONFIG.log_file}", []

    answer_md = "### 回答\n" + result.answer
    rows = []
    for rank, item in enumerate(result.retrieved, start=1):
        page = item.chunk.metadata.get("page", "-")
        preview = item.chunk.text[:120].replace("\n", " ")
        rows.append(
            [
                rank,
                item.chunk.source_name,
                page,
                f"{item.retrieval_score:.4f}",
                f"{item.rerank_score:.4f}" if item.rerank_score is not None else "-",
                preview,
            ]
        )
    return answer_md, rows


def clear_index():
    PIPELINE.clear()
    LOGGER.info("知识库索引已清空。")
    return "知识库已清空。", []


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=CONFIG.project_name) as demo:
        gr.Markdown(
            """
# 本地中文 RAG Demo

上传 PDF / 文本文件，系统会依次完成：**切块 -> embedding 检索 -> reranker 重排 -> 本地中文大模型生成**。

> 提示：首次构建索引和首次提问会下载/加载模型，耗时会明显更长。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="上传知识库文件",
                    file_count="multiple",
                    file_types=SUPPORTED_FILE_TYPES,
                    type="filepath",
                )
                build_button = gr.Button("建立索引", variant="primary")
                clear_button = gr.Button("清空索引")
                build_status = gr.Markdown("尚未建立索引。")
                source_table = gr.Dataframe(
                    headers=["序号", "来源文件"],
                    value=[],
                    datatype=["number", "str"],
                    wrap=True,
                    label="当前来源文件",
                    interactive=False,
                )

            with gr.Column(scale=2):
                question = gr.Textbox(
                    label="输入问题",
                    placeholder="例如：出差报销要在多久之内提交？需要哪些材料？",
                    lines=3,
                )
                with gr.Row():
                    retrieve_top_k = gr.Slider(2, 12, value=CONFIG.retrieve_top_k, step=1, label="召回 top-k")
                    rerank_top_k = gr.Slider(1, 6, value=CONFIG.rerank_top_k, step=1, label="重排后保留")
                ask_button = gr.Button("开始提问", variant="primary")
                answer_output = gr.Markdown("### 回答\n等待提问。")
                retrieved_table = gr.Dataframe(
                    headers=["排名", "来源", "页码", "retrieval", "rerank", "文本预览"],
                    value=[],
                    datatype=["number", "str", "str", "str", "str", "str"],
                    wrap=True,
                    label="最终送入生成模型的上下文",
                    interactive=False,
                )

        gr.Examples(
            examples=[
                ["出差报销要在多久之内提交？需要准备什么材料？"],
                ["首次使用 VPN 应该怎么申请？"],
                ["请假 2 天由谁审批？"],
            ],
            inputs=[question],
        )

        build_button.click(
            fn=build_index,
            inputs=[file_input],
            outputs=[build_status, source_table],
        )
        ask_button.click(
            fn=answer_question,
            inputs=[question, retrieve_top_k, rerank_top_k],
            outputs=[answer_output, retrieved_table],
        )
        clear_button.click(
            fn=clear_index,
            inputs=None,
            outputs=[build_status, source_table],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
