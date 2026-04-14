from __future__ import annotations

from .schemas import RetrievedChunk


SYSTEM_PROMPT = """你是一个严谨的中文知识库问答助手。
请严格遵守以下规则：
1. 只能依据给定的参考资料作答，不要引入未给出的事实。
2. 如果参考资料不足以回答问题，请明确说“根据当前知识库，暂无足够依据回答这个问题”。
3. 回答尽量简洁、清晰、中文自然。
4. 在使用某条证据的句子后标注引用编号，例如 [1]、[2]。
5. 回答末尾补充“参考来源”小节，列出用到的来源。
"""


def build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    context_blocks: list[str] = []
    for index, item in enumerate(contexts, start=1):
        page = item.chunk.metadata.get("page")
        page_desc = f"，第 {page} 页" if page else ""
        header = f"[{index}] 来源：{item.chunk.source_name}{page_desc}"
        context_blocks.append(f"{header}\n{item.chunk.text}")

    joined_context = "\n\n".join(context_blocks)
    return (
        f"# 参考资料\n{joined_context}\n\n"
        f"# 用户问题\n{question}\n\n"
        "# 输出要求\n"
        "- 先给出直接答案。\n"
        "- 然后给出“参考来源”小节。\n"
        "- 若证据不足，请直接说明。\n"
    )
