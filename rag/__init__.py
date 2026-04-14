from __future__ import annotations

__all__ = ["ModularRAGPipeline"]


def __getattr__(name: str):
    if name == "ModularRAGPipeline":
        from .pipeline import ModularRAGPipeline

        return ModularRAGPipeline
    raise AttributeError(name)
