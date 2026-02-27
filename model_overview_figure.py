#!/usr/bin/env python3
"""
Create a multi-panel model-overview figure from pipeline forecast outputs.

This is a data-driven Python figure (matplotlib) arranged to match the conceptual
layout: kernel, stock transition heatmap, arrivals transition heatmap, side totals,
and summary daily conversion chart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_lib.overview import create_figure, infer_asof_tag, load_inputs


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
