from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


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

    return {k: pd.read_csv(v) for k, v in files.items()}
