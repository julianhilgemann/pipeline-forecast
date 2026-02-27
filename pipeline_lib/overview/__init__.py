from __future__ import annotations

from .io import infer_asof_tag, load_inputs

__all__ = ["create_figure", "infer_asof_tag", "load_inputs"]


def __getattr__(name: str):
    if name == "create_figure":
        from .figure import create_figure

        return create_figure
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
