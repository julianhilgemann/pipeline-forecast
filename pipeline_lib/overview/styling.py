from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def _fmt_scale_value(v: float) -> str:
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def add_horizontal_inset_scale(
    ax: plt.Axes,
    cmap: LinearSegmentedColormap,
    vmax: float,
    corner: str,
    label: str,
) -> None:
    if vmax <= 0:
        return

    if corner == "top_right":
        rect = [0.56, 0.80, 0.36, 0.12]
    elif corner == "bottom_left":
        rect = [0.08, 0.10, 0.36, 0.12]
    else:
        rect = [0.56, 0.80, 0.36, 0.12]

    cax = ax.inset_axes(rect)
    grad = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    cax.imshow(
        grad,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        extent=[0.0, vmax, 0.0, 1.0],
    )
    cax.set_yticks([])
    cax.set_xticks([0.0, vmax * 0.5, vmax])
    cax.set_xticklabels(
        [_fmt_scale_value(0.0), _fmt_scale_value(vmax * 0.5), _fmt_scale_value(vmax)],
        fontsize=6,
        color="#111827",
    )
    cax.tick_params(axis="x", length=2, colors="#111827", pad=1)
    for spine in cax.spines.values():
        spine.set_color("#94A3B8")
        spine.set_linewidth(0.8)
    cax.set_title(label, fontsize=9, color="#111827", pad=2, loc="left")


def style_panel(
    ax: plt.Axes,
    title: str,
    accent: str,
    title_size: int = 9,
    grid_axis: str | None = None,
) -> None:
    axis_txt = "#111827"
    border = "#94A3B8"
    grid = "#D1D5DB"
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_color(border)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=axis_txt, labelsize=8)
    ax.set_title(title, color="#0F172A", fontsize=title_size, fontweight="bold", loc="left", pad=6)
    if grid_axis:
        ax.grid(axis=grid_axis, color=grid, linewidth=0.8, alpha=0.9)
    else:
        ax.grid(False)
    ax.xaxis.label.set_color(axis_txt)
    ax.yaxis.label.set_color(axis_txt)
    ax.title.set_color("#111827")
