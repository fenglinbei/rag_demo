from __future__ import annotations

from pathlib import Path


def ensure_model_path(model_name_or_path: str, role: str) -> None:
    """Fail fast when a local model path is configured but missing."""
    model_ref = model_name_or_path.strip()
    if not model_ref:
        raise ValueError(f"{role} 模型配置为空，请检查环境变量。")

    candidate = Path(model_ref).expanduser()
    looks_like_local = candidate.is_absolute() or model_ref.startswith(".")
    if looks_like_local and not candidate.exists():
        raise FileNotFoundError(
            f"{role} 模型路径不存在：{model_ref}。"
            "请先下载模型，或把配置改为 HuggingFace/ModelScope 的模型名。"
        )
