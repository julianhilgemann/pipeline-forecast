from __future__ import annotations

from .config import Config

__all__ = ["Config", "run_pipeline"]


def __getattr__(name: str):
    if name == "run_pipeline":
        from .pipeline_runner import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
