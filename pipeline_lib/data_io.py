from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def surface_to_dataframe(surface: np.ndarray, max_age: int, horizon: int) -> pd.DataFrame:
    out = pd.DataFrame(surface)
    out.insert(0, "age_biz_days", np.arange(max_age + 1, dtype=int))
    out.columns = ["age_biz_days"] + [f"offset_{i}" for i in range(1, horizon + 1)]
    return out


def write_single_forecast_outputs(
    out_dir: Path,
    asof_tag: str,
    result: dict[str, object],
    cfg: Config,
) -> None:
    result["active_age"].to_csv(out_dir / f"active_stock_by_age_{asof_tag}.csv", index=False)
    result["kernel"].to_csv(out_dir / f"win_kernel_{asof_tag}.csv", index=False)
    result["arrivals_fc"].to_csv(out_dir / f"forecast_arrivals_by_day_{asof_tag}.csv", index=False)
    result["pred_by_day"].to_csv(out_dir / f"predicted_wins_by_day_{asof_tag}.csv", index=False)
    result["age_error"].to_csv(out_dir / f"age_error_{asof_tag}.csv", index=False)

    stock_surface = result["stock_surface"]
    actual_stock_surface = result["actual_stock_surface"]

    stock_surface_df = surface_to_dataframe(stock_surface, cfg.max_age_bdays, cfg.horizon_bdays)
    stock_surface_df.to_csv(out_dir / f"stock_surface_{asof_tag}.csv", index=False)

    actual_surface_df = surface_to_dataframe(actual_stock_surface, cfg.max_age_bdays, cfg.horizon_bdays)
    actual_surface_df.to_csv(out_dir / f"actual_stock_surface_{asof_tag}.csv", index=False)

    error_surface = stock_surface - actual_stock_surface
    error_surface_df = surface_to_dataframe(error_surface, cfg.max_age_bdays, cfg.horizon_bdays)
    error_surface_df.to_csv(out_dir / f"stock_error_surface_{asof_tag}.csv", index=False)
