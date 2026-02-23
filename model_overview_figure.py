#!/usr/bin/env python3
"""
Create a multi-panel model-overview figure from pipeline forecast outputs.

This is a data-driven Python figure (matplotlib) arranged to match the conceptual
layout: kernel, stock transition heatmap, arrivals transition heatmap, side totals,
and summary daily conversion chart.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def infer_asof_tag(output_dir: Path) -> str:
    pattern = re.compile(r"predicted_wins_by_day_(\d{4}-\d{2}-\d{2})\.csv$")
    tags: list[str] = []
    for p in output_dir.glob("predicted_wins_by_day_*.csv"):
        m = pattern.match(p.name)
        if m:
            tags.append(m.group(1))
    if not tags:
        raise FileNotFoundError(
            f"No predicted_wins_by_day_YYYY-MM-DD.csv found in {output_dir}"
        )
    return sorted(tags)[-1]


def load_inputs(output_dir: Path, asof_tag: str) -> dict[str, pd.DataFrame]:
    files = {
        "kernel": output_dir / f"win_kernel_{asof_tag}.csv",
        "active_age": output_dir / f"active_stock_by_age_{asof_tag}.csv",
        "arrivals_fc": output_dir / f"forecast_arrivals_by_day_{asof_tag}.csv",
        "pred_by_day": output_dir / f"predicted_wins_by_day_{asof_tag}.csv",
        "stock_surface": output_dir / f"stock_surface_{asof_tag}.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required CSVs:\n" + "\n".join(missing))

    data = {k: pd.read_csv(v) for k, v in files.items()}
    return data


def build_arrivals_surface(
    arrivals_fc: pd.DataFrame, kernel: pd.DataFrame, horizon: int
) -> np.ndarray:
    win_mass = kernel["win_kernel_mass"].to_numpy(dtype=float)
    arrivals = (
        arrivals_fc.sort_values("day_offset")["forecast_arrivals"]
        .to_numpy(dtype=float)[:horizon]
    )
    mat = np.zeros((horizon, horizon), dtype=float)
    for create_ix in range(horizon):
        n = arrivals[create_ix]
        for close_ix in range(create_ix, horizon):
            age = close_ix - create_ix
            if age >= len(win_mass):
                break
            mat[create_ix, close_ix] = n * win_mass[age]
    return mat


def smooth_series(values: np.ndarray, window: int = 9, sigma: float = 2.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, arr.size if arr.size % 2 == 1 else max(3, arr.size - 1))
    if window < 3:
        return arr.copy()

    radius = window // 2
    x = np.arange(-radius, radius + 1, dtype=float)
    weights = np.exp(-(x * x) / (2.0 * sigma * sigma))
    weights = weights / weights.sum()
    padded = np.pad(arr, (radius, radius), mode="edge")
    smoothed = np.convolve(padded, weights, mode="valid")
    return smoothed


def smooth_matrix(values: np.ndarray, passes: int = 1) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return arr.copy()
    out = arr.copy()
    for _ in range(max(1, passes)):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            padded[:-2, :-2]
            + 2.0 * padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2]
            + 4.0 * padded[1:-1, 1:-1]
            + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2]
            + 2.0 * padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 16.0
    return out


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
    for s in cax.spines.values():
        s.set_color("#94A3B8")
        s.set_linewidth(0.8)
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
    for s in ax.spines.values():
        s.set_color(border)
        s.set_linewidth(1.0)
    ax.tick_params(colors=axis_txt, labelsize=8)
    ax.set_title(title, color="#0F172A", fontsize=title_size, fontweight="bold", loc="left", pad=6)
    if grid_axis:
        ax.grid(axis=grid_axis, color=grid, linewidth=0.8, alpha=0.9)
    else:
        ax.grid(False)
    ax.xaxis.label.set_color(axis_txt)
    ax.yaxis.label.set_color(axis_txt)
    ax.title.set_color("#111827")


def create_figure(
    asof_tag: str,
    kernel: pd.DataFrame,
    active_age: pd.DataFrame,
    arrivals_fc: pd.DataFrame,
    pred_by_day: pd.DataFrame,
    stock_surface_df: pd.DataFrame,
    save_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#CBD5E1",
        }
    )

    colors = {
        "text": "#111827",
        "muted": "#374151",
        "border": "#94A3B8",
        "grid": "#D1D5DB",
        "kernel": "#D97706",
        "stock": "#B5475B",
        "arrivals": "#2B6CB0",
        "summary": "#0F766E",
    }

    age_col = "age_biz_days"
    stock_surface = (
        stock_surface_df.sort_values(age_col).drop(columns=[age_col]).to_numpy(dtype=float)
    )
    horizon = stock_surface.shape[1]
    arrivals_surface = build_arrivals_surface(arrivals_fc, kernel, horizon)
    stock_surface_display = smooth_matrix(stock_surface, passes=1)
    arrivals_surface_display = smooth_matrix(arrivals_surface, passes=1)

    pred = pred_by_day.sort_values("day_offset").head(horizon).copy()
    offset_axis = np.arange(horizon, dtype=int)

    age_axis = np.arange(stock_surface.shape[0], dtype=int)
    active_cnt = np.zeros_like(age_axis, dtype=float)
    active = active_age.sort_values("age_biz_days")
    for r in active.itertuples(index=False):
        age = int(r.age_biz_days)
        if 0 <= age < len(active_cnt):
            active_cnt[age] = float(r.active_count)
    stock_total_by_age = stock_surface.sum(axis=1)
    arrivals_vec = (
        arrivals_fc.sort_values("day_offset")["forecast_arrivals"].to_numpy(dtype=float)[:horizon]
    )
    arrivals_total_by_origin = arrivals_surface.sum(axis=1)
    total_expected = pred["expected_wins_stock"].to_numpy(dtype=float) + pred[
        "expected_wins_arrivals"
    ].to_numpy(dtype=float)
    total_expected_smoothed = smooth_series(total_expected, window=7, sigma=1.6)

    kernel_age = kernel["age_biz_days"].to_numpy(dtype=float)
    kernel_mass = kernel["win_kernel_mass"].to_numpy(dtype=float)
    kernel_smooth = smooth_series(kernel_mass, window=11, sigma=2.2)
    if kernel_smooth.sum() > 0 and kernel_mass.sum() > 0:
        kernel_smooth = kernel_smooth * (kernel_mass.sum() / kernel_smooth.sum())

    stock_cmap = LinearSegmentedColormap.from_list(
        "stock_cmap", ["#FFFFFF", "#FBECEE", "#F2C7CF", "#D98A99", "#B5475B", "#8A3042"]
    )
    arrivals_cmap = LinearSegmentedColormap.from_list(
        "arrivals_cmap", ["#FFFFFF", "#EAF2FB", "#C9DFF5", "#95BEE8", "#5C97D3", "#2B6CB0"]
    )

    vmax1 = (
        float(np.percentile(stock_surface_display, 99))
        if np.any(stock_surface_display > 0)
        else 1.0
    )
    vmax1 = max(vmax1, 1e-6)
    vmax2 = (
        float(np.percentile(arrivals_surface_display, 99))
        if np.any(arrivals_surface_display > 0)
        else 1.0
    )
    vmax2 = max(vmax2, 1e-6)

    fig = plt.figure(figsize=(15, 14.2), facecolor="white")
    gs = GridSpec(
        4,
        3,
        figure=fig,
        width_ratios=[0.20, 0.62, 0.18],
        height_ratios=[0.32, 0.34, 0.34, 0.32],
        wspace=0.18,
        hspace=0.50,
        left=0.12,
        right=0.93,
        top=0.87,
        bottom=0.10,
    )

    # Top center: smoothed decision kernel.
    ax_kernel = fig.add_subplot(gs[0, 1])
    style_panel(
        ax_kernel,
        "Positive Decision Kernel (Smoothed)\nWin probability mass by decision age",
        colors["kernel"],
        title_size=9,
        grid_axis="y",
    )
    ax_kernel.plot(kernel_age, kernel_smooth, color=colors["kernel"], linewidth=1.2)
    ax_kernel.fill_between(
        kernel_age,
        0,
        kernel_smooth,
        color=colors["kernel"],
        alpha=0.16,
    )
    ax_kernel.set_xlim(0, min(int(kernel["age_biz_days"].max()), 90))
    kmax = float(np.max(kernel_smooth)) if np.size(kernel_smooth) else 0.0
    ax_kernel.set_ylim(0.0, max(1e-6, 1.2 * kmax))
    ax_kernel.margins(x=0.01)
    ax_kernel.set_xlabel("Decision age [business days]", fontsize=9, labelpad=1)
    ax_kernel.set_ylabel("P(win at age) [probability/day]", fontsize=9)

    # Middle row stock heatmap + aligned side bars.
    ax_stock_heat = fig.add_subplot(gs[1, 1])
    style_panel(
        ax_stock_heat,
        "Existing Pipeline Convolution Heatmap",
        colors["stock"],
        title_size=9,
    )
    ax_stock_heat.imshow(
        stock_surface_display,
        origin="lower",
        aspect="auto",
        cmap=stock_cmap,
        interpolation="nearest",
        extent=[-0.5, horizon - 0.5, -0.5, len(age_axis) - 0.5],
        vmin=0.0,
        vmax=vmax1,
    )
    ax_stock_heat.set_xlabel(
        "Future day offset [business days; 0 = first forecast day]", fontsize=9, labelpad=1
    )
    ax_stock_heat.set_ylabel("Pipeline age [business days; 0 = as-of]", fontsize=9)
    ax_stock_heat.set_xticks(np.arange(0, horizon, 5))
    ax_stock_heat.set_yticks(np.arange(0, len(age_axis), 10))
    add_horizontal_inset_scale(
        ax=ax_stock_heat,
        cmap=stock_cmap,
        vmax=vmax1,
        corner="top_right",
        label="Expected wins [deals/day]",
    )

    ax_stock_left = fig.add_subplot(gs[1, 0], sharey=ax_stock_heat)
    style_panel(
        ax_stock_left,
        "Pipeline Potential by Age",
        colors["stock"],
        title_size=9,
        grid_axis="x",
    )
    ax_stock_left.barh(age_axis, active_cnt, height=0.86, color=colors["stock"], alpha=0.82, edgecolor="none")
    ax_stock_left.set_xlabel("Open opps [count]", fontsize=8)
    ax_stock_left.set_ylabel("Age [bd]", fontsize=8)
    ax_stock_left.invert_xaxis()
    ax_stock_left.set_ylim(ax_stock_heat.get_ylim())
    ax_stock_left.tick_params(axis="y", labelleft=False)

    ax_stock_right = fig.add_subplot(gs[1, 2], sharey=ax_stock_heat)
    style_panel(
        ax_stock_right,
        "Total Conversion per Age",
        colors["stock"],
        title_size=9,
        grid_axis="x",
    )
    ax_stock_right.barh(
        age_axis, stock_total_by_age, height=0.86, color=colors["stock"], alpha=0.82, edgecolor="none"
    )
    ax_stock_right.set_xlabel("Expected wins [deals]", fontsize=8)
    ax_stock_right.set_ylabel("Age [bd]", fontsize=8)
    ax_stock_right.set_ylim(ax_stock_heat.get_ylim())
    ax_stock_right.tick_params(axis="y", labelleft=False)

    # Bottom row arrivals heatmap + aligned side bars.
    ax_arr_heat = fig.add_subplot(gs[2, 1])
    style_panel(
        ax_arr_heat,
        "Future Arrivals Convolution Heatmap",
        colors["arrivals"],
        title_size=9,
    )
    ax_arr_heat.imshow(
        arrivals_surface_display,
        origin="lower",
        aspect="auto",
        cmap=arrivals_cmap,
        interpolation="nearest",
        extent=[-0.5, horizon - 0.5, -0.5, horizon - 0.5],
        vmin=0.0,
        vmax=vmax2,
    )
    ax_arr_heat.set_xlabel(
        "Future conversion day offset [business days; 0 = first forecast day]",
        fontsize=9,
        labelpad=1,
    )
    ax_arr_heat.set_ylabel("Arrival day offset [business days; 0 at top]", fontsize=9)
    ax_arr_heat.set_xticks(np.arange(0, horizon, 5))
    ax_arr_heat.set_yticks(np.arange(0, horizon, 5))
    # Force ascending offset order from top to bottom across the arrivals stack.
    ax_arr_heat.set_ylim(horizon - 0.5, -0.5)
    add_horizontal_inset_scale(
        ax=ax_arr_heat,
        cmap=arrivals_cmap,
        vmax=vmax2,
        corner="bottom_left",
        label="Expected wins [deals/day]",
    )

    ax_arr_left = fig.add_subplot(gs[2, 0], sharey=ax_arr_heat)
    style_panel(
        ax_arr_left,
        "SARIMAX / Seasonal-Naive Forecast",
        colors["arrivals"],
        title_size=9,
        grid_axis="x",
    )
    ax_arr_left.barh(
        offset_axis, arrivals_vec, height=0.86, color=colors["arrivals"], alpha=0.85, edgecolor="none"
    )
    ax_arr_left.set_xlabel("Forecast arrivals [opps/day]", fontsize=8)
    ax_arr_left.set_ylabel("Arrival offset [bd; 0 top]", fontsize=8)
    ax_arr_left.invert_xaxis()
    ax_arr_left.set_ylim(ax_arr_heat.get_ylim())
    ax_arr_left.tick_params(axis="y", labelleft=False)

    ax_arr_right = fig.add_subplot(gs[2, 2], sharey=ax_arr_heat)
    style_panel(
        ax_arr_right,
        "Total Conversion of Arrivals",
        colors["arrivals"],
        title_size=9,
        grid_axis="x",
    )
    ax_arr_right.barh(
        offset_axis,
        arrivals_total_by_origin,
        height=0.86,
        color=colors["arrivals"],
        alpha=0.85,
        edgecolor="none",
    )
    ax_arr_right.set_xlabel("Expected wins [deals]", fontsize=8)
    ax_arr_right.set_ylabel("Arrival offset [bd; 0 top]", fontsize=8)
    ax_arr_right.set_ylim(ax_arr_heat.get_ylim())
    ax_arr_right.tick_params(axis="y", labelleft=False)

    # Bottom summary chart (predicted only; no actual wins).
    ax_sum = fig.add_subplot(gs[3, 1])
    style_panel(
        ax_sum,
        "Summary: Daily Expected Total Conversion",
        colors["summary"],
        title_size=10,
        grid_axis="y",
    )
    stock_vals = pred["expected_wins_stock"].to_numpy(dtype=float)
    arr_vals = pred["expected_wins_arrivals"].to_numpy(dtype=float)
    ax_sum.bar(
        offset_axis, stock_vals, color=colors["stock"], alpha=0.85, label="From existing stock"
    )
    ax_sum.bar(
        offset_axis,
        arr_vals,
        bottom=stock_vals,
        color=colors["arrivals"],
        alpha=0.85,
        label="From forecast arrivals",
    )
    ax_sum.plot(
        offset_axis,
        total_expected_smoothed,
        color=colors["summary"],
        linewidth=2.2,
        label="Smoothed expected total",
    )
    summary_max = max(
        float(np.max(stock_vals + arr_vals)) if stock_vals.size else 0.0,
        float(np.max(total_expected_smoothed)) if total_expected_smoothed.size else 0.0,
    )
    summary_ymax = max(1e-6, 1.2 * summary_max)
    ax_sum.set_ylim(0.0, summary_ymax)
    label_pad = 0.02 * summary_ymax
    for i in range(min(len(offset_axis), len(arrivals_vec), len(total_expected_smoothed))):
        y_lab = min(total_expected_smoothed[i] + label_pad, 0.98 * summary_ymax)
        ax_sum.text(
            offset_axis[i],
            y_lab,
            f"{int(round(arrivals_vec[i]))}",
            ha="center",
            va="bottom",
            fontsize=6,
            color=colors["muted"],
        )
    ax_sum.set_xlabel(
        "Future decision day offset [business days, 0 = first forecast day]",
        fontsize=9,
        labelpad=1,
    )
    ax_sum.set_ylabel("Expected wins [deals/day]", fontsize=9)
    ax_sum.set_xlim(-0.6, horizon - 0.4)
    ax_sum.set_xticks(np.arange(0, horizon, 5))
    ax_sum.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_sum.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    leg = ax_sum.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.38),
        frameon=False,
        fontsize=8,
        ncol=3,
    )
    for text in leg.get_texts():
        text.set_color(colors["text"])

    # Hide unused corner cells.
    for pos in [(0, 0), (0, 2), (3, 0), (3, 2)]:
        ax = fig.add_subplot(gs[pos])
        ax.set_axis_off()

    # Figure-level framing and directional annotations.
    fig.text(
        0.5,
        0.982,
        f"Business-Day Conversion Forecast Architecture | As-of {asof_tag}",
        ha="center",
        va="top",
        color=colors["text"],
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.964,
        "Decision-kernel convolution of existing pipeline and forecasted arrivals",
        ha="center",
        va="top",
        color=colors["muted"],
        fontsize=9,
    )
    fig.patches.append(
        plt.Rectangle(
            (0.01, 0.01),
            0.98,
            0.98,
            transform=fig.transFigure,
            fill=False,
            edgecolor=colors["border"],
            linewidth=1.2,
        )
    )
    fig.savefig(save_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render model-overview multi-panel figure")
    p.add_argument("--output-dir", default="outputs", help="Folder with forecast CSV outputs")
    p.add_argument("--as-of-date", default=None, help="As-of date tag YYYY-MM-DD")
    p.add_argument(
        "--save-path",
        default=None,
        help="Optional output file path. Defaults to outputs/figures/model_overview_asof_<date>.png",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    asof_tag = args.as_of_date if args.as_of_date else infer_asof_tag(out_dir)
    data = load_inputs(out_dir, asof_tag)

    save_path = (
        Path(args.save_path)
        if args.save_path
        else out_dir / "figures" / f"model_overview_asof_{asof_tag}.png"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    create_figure(
        asof_tag=asof_tag,
        kernel=data["kernel"],
        active_age=data["active_age"],
        arrivals_fc=data["arrivals_fc"],
        pred_by_day=data["pred_by_day"],
        stock_surface_df=data["stock_surface"],
        save_path=save_path,
    )
    print(f"Saved figure: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
