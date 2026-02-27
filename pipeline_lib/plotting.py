from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_heatmap_ppm(surface: np.ndarray, path: Path, cell_px: int = 6) -> None:
    if surface.size == 0:
        return

    val = surface.astype(float)
    vmax = float(val.max()) if np.isfinite(val).any() else 1.0
    if vmax <= 0:
        norm = np.zeros_like(val)
    else:
        norm = np.clip(val / vmax, 0.0, 1.0)

    # Warm gradient: very light yellow -> orange -> red.
    def color(v: float) -> tuple[int, int, int]:
        r = 255
        g = int(250 - 170 * v)
        b = int(220 - 220 * v)
        return (r, max(0, min(255, g)), max(0, min(255, b)))

    h_cells, w_cells = norm.shape
    w = w_cells * cell_px
    h = h_cells * cell_px
    out = bytearray()
    for y in range(h):
        cy = h_cells - 1 - (y // cell_px)
        for x in range(w):
            cx = x // cell_px
            out.extend(color(float(norm[cy, cx])))

    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(out)


def render_plots(
    output_dir: Path,
    as_of_date: str,
    stock_surface: np.ndarray,
    pred_by_day: pd.DataFrame,
) -> dict[str, Path]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    heatmap_png = fig_dir / f"heatmap_asof_{as_of_date}.png"
    line_png = fig_dir / f"wins_line_asof_{as_of_date}.png"
    fallback_heatmap = fig_dir / f"heatmap_asof_{as_of_date}.ppm"

    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(11, 7))
        im = ax.imshow(stock_surface, aspect="auto", origin="lower", cmap="YlOrRd")
        ax.set_title(f"Expected Wins from Active Stock Surface (as-of {as_of_date})")
        ax.set_xlabel("Future Business Day Offset")
        ax.set_ylabel("Current Age Bucket (Business Days)")
        fig.colorbar(im, ax=ax, label="Expected Wins")
        fig.tight_layout()
        fig.savefig(heatmap_png, dpi=150)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(11, 4))
        ax2.plot(pred_by_day["day_offset"], pred_by_day["expected_wins_total"], label="Predicted")
        ax2.plot(pred_by_day["day_offset"], pred_by_day["actual_wins"], label="Actual")
        ax2.set_title(f"Predicted vs Actual Won Deals by Day (as-of {as_of_date})")
        ax2.set_xlabel("Future Business Day Offset")
        ax2.set_ylabel("Won Count")
        ax2.grid(alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(line_png, dpi=150)
        plt.close(fig2)
        return {"heatmap": heatmap_png, "line": line_png}
    except Exception:
        save_heatmap_ppm(stock_surface, fallback_heatmap)
        return {"heatmap": fallback_heatmap}


def render_structure_chart_svg(output_dir: Path) -> Path:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    chart_path = fig_dir / "forecast_structure_chart.svg"

    width = 1100
    height = 1500
    bg = "#0b0f12"
    border = "#a0a7af"
    green = "#16a34a"
    red = "#e57373"
    blue = "#42a5f5"

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def rect(x: int, y: int, w: int, h: int, color: str, sw: int = 3) -> str:
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="none" stroke="{color}" stroke-width="{sw}"/>'
        )

    def line(x1: int, y1: int, x2: int, y2: int, color: str, marker: str, sw: int = 4) -> str:
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}" marker-end="url(#{marker})"/>'
        )

    def multiline_text(
        x: int,
        y: int,
        lines: list[str],
        color: str,
        size: int = 32,
        line_h: int = 42,
        weight: str = "600",
        anchor: str = "middle",
    ) -> str:
        out = [
            f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" '
            f'text-anchor="{anchor}">'
        ]
        for i, ln in enumerate(lines):
            dy = 0 if i == 0 else line_h
            out.append(f'<tspan x="{x}" dy="{dy}">{esc(ln)}</tspan>')
        out.append("</text>")
        return "".join(out)

    def rotated_multiline_text(
        cx: int,
        cy: int,
        lines: list[str],
        color: str,
        size: int = 31,
        line_h: int = 40,
        weight: str = "600",
        angle: int = -90,
    ) -> str:
        out = [f'<g transform="rotate({angle} {cx} {cy})">']
        out.append(
            f'<text x="{cx}" y="{cy}" fill="{color}" font-size="{size}" '
            f'font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" '
            f'text-anchor="middle">'
        )
        for i, ln in enumerate(lines):
            dy = 0 if i == 0 else line_h
            out.append(f'<tspan x="{cx}" dy="{dy}">{esc(ln)}</tspan>')
        out.append("</text></g>")
        return "".join(out)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>',
        "<defs>",
        f'<marker id="arrow-green" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="{green}"/></marker>',
        f'<marker id="arrow-red" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="{red}"/></marker>',
        f'<marker id="arrow-blue" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="{blue}"/></marker>',
        "</defs>",
        rect(28, 28, width - 56, height - 56, border, 4),
        multiline_text(550, 135, ["Future Conversion"], green, size=36, line_h=40, weight="700"),
        line(320, 165, 790, 165, green, "arrow-green", 4),
        rect(320, 225, 460, 130, green, 3),
        multiline_text(
            550,
            260,
            ["Positive Decision Kernel", "Probability Density Plot", "Conversion at Age"],
            green,
            size=34,
            line_h=40,
            weight="700",
        ),
        line(125, 860, 125, 390, red, "arrow-red", 4),
        rotated_multiline_text(85, 640, ["Time into the Past"], red, size=33, line_h=38, weight="700"),
        rect(180, 390, 120, 470, red, 3),
        rect(320, 390, 460, 470, red, 3),
        rect(820, 390, 160, 470, red, 3),
        rotated_multiline_text(
            240,
            630,
            ["Pipeline Potential at age"],
            red,
            size=33,
            line_h=38,
            weight="700",
        ),
        multiline_text(
            550,
            520,
            [
                "Transition Matrix",
                "Conversion Probability",
                "conditional",
                "on age,",
                "convolved with decision kernel",
                "",
                "HEATMAP",
            ],
            red,
            size=38,
            line_h=46,
            weight="700",
        ),
        rotated_multiline_text(
            900,
            612,
            ["Total Conversion per Age", "from existing pipeline"],
            red,
            size=32,
            line_h=38,
            weight="700",
        ),
        line(125, 900, 125, 1380, blue, "arrow-blue", 4),
        rotated_multiline_text(85, 1140, ["Future time"], blue, size=33, line_h=38, weight="700"),
        rect(180, 900, 120, 400, blue, 3),
        rect(320, 900, 460, 400, blue, 3),
        rect(820, 900, 160, 400, blue, 3),
        rotated_multiline_text(
            240,
            1102,
            ["Future Daily Arrivals (SARIMAX)"],
            blue,
            size=33,
            line_h=38,
            weight="700",
        ),
        multiline_text(
            550,
            1005,
            [
                "Transition Matrix",
                "Conversion Probability",
                "of new arrivals convolved",
                "with decision kernel",
                "",
                "HEATMAP",
            ],
            blue,
            size=37,
            line_h=45,
            weight="700",
        ),
        rotated_multiline_text(
            900,
            1085,
            ["Total Conversion of arrivals"],
            blue,
            size=33,
            line_h=38,
            weight="700",
        ),
        rect(335, 1320, 430, 110, green, 3),
        multiline_text(
            550,
            1370,
            ["Summary Barchart of daily", "total conversion"],
            green,
            size=34,
            line_h=40,
            weight="700",
        ),
        "</svg>",
    ]

    chart_path.write_text("\n".join(svg_parts), encoding="utf-8")
    return chart_path
