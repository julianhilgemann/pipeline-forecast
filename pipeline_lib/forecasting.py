from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .calendar import forecast_business_day_range
from .config import Config
from .modeling import (
    daily_metrics,
    estimate_kernel_from_training,
    project_wins_from_active_stock,
    project_wins_from_arrivals,
    seasonal_naive_arrival_forecast,
)
from .warehouse import (
    active_opportunities_asof,
    active_snapshot_asof,
    actual_wins_by_day,
    cohort_actual_surface,
    training_window_extract,
)


def single_asof_forecast(
    con: duckdb.DuckDBPyConnection,
    cal: pd.DataFrame,
    as_of_date: str,
    cfg: Config,
) -> dict[str, object]:
    as_of_row = cal[cal["date"] == pd.Timestamp(as_of_date).date()]
    if as_of_row.empty or not bool(as_of_row.iloc[0]["is_business_day"]):
        raise ValueError(f"as_of_date must be a business day in calendar: {as_of_date}")
    as_of_idx = int(as_of_row.iloc[0]["biz_day_index"])
    train_start_idx = as_of_idx - cfg.training_window_bdays + 1
    train_end_idx = as_of_idx

    active_age = active_snapshot_asof(con, as_of_date, as_of_idx, cfg.max_age_bdays)
    active_opp = active_opportunities_asof(con, as_of_date, as_of_idx, cfg.max_age_bdays)
    train_df = training_window_extract(con, train_start_idx, train_end_idx)
    kernel = estimate_kernel_from_training(
        train_df, train_start_idx, train_end_idx, cfg.max_age_bdays
    )

    future_days = forecast_business_day_range(cal, as_of_idx, cfg.horizon_bdays)
    if len(future_days) < cfg.horizon_bdays:
        raise ValueError(
            "Not enough future business days for horizon. Extend end_date or reduce horizon."
        )

    stock_by_day, stock_surface = project_wins_from_active_stock(
        active_age, kernel, cfg.horizon_bdays, cfg.max_age_bdays
    )
    arrivals_fc = seasonal_naive_arrival_forecast(con, as_of_date, future_days)
    arrivals_wins = project_wins_from_arrivals(
        arrivals_fc, kernel, cfg.horizon_bdays, cfg.max_age_bdays
    )

    actual = actual_wins_by_day(con, as_of_idx, cfg.horizon_bdays)
    pred = (
        future_days[["date", "biz_day_index", "day_offset"]]
        .merge(stock_by_day, on="day_offset", how="left")
        .merge(arrivals_wins, on="day_offset", how="left")
        .merge(actual, on="biz_day_index", how="left")
        .fillna({"expected_wins_stock": 0.0, "expected_wins_arrivals": 0.0, "actual_wins": 0.0})
    )
    pred["expected_wins_total"] = pred["expected_wins_stock"] + pred["expected_wins_arrivals"]

    metrics = daily_metrics(
        pred["expected_wins_total"].to_numpy(dtype=float),
        pred["actual_wins"].to_numpy(dtype=float),
    )

    actual_stock_surface = cohort_actual_surface(
        con, active_opp, as_of_idx, cfg.horizon_bdays, cfg.max_age_bdays
    )
    age_pred = stock_surface.sum(axis=1)
    age_actual = actual_stock_surface.sum(axis=1)
    age_err = pd.DataFrame(
        {
            "age_biz_days": np.arange(cfg.max_age_bdays + 1, dtype=int),
            "predicted_wins_from_stock": age_pred,
            "actual_wins_from_stock_cohort": age_actual,
            "error": age_pred - age_actual,
            "abs_error": np.abs(age_pred - age_actual),
        }
    )

    return {
        "as_of_date": as_of_date,
        "as_of_idx": as_of_idx,
        "active_age": active_age,
        "kernel": kernel,
        "future_days": future_days,
        "arrivals_fc": arrivals_fc,
        "pred_by_day": pred,
        "stock_surface": stock_surface,
        "actual_stock_surface": actual_stock_surface,
        "age_error": age_err,
        "metrics": metrics,
    }


def rolling_backtest(
    con: duckdb.DuckDBPyConnection, cal: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    biz = cal[cal["is_business_day"]].sort_values("biz_day_index").copy()
    min_idx = cfg.training_window_bdays + 5
    max_idx = int(biz["biz_day_index"].max()) - cfg.horizon_bdays
    candidates = biz[(biz["biz_day_index"] >= min_idx) & (biz["biz_day_index"] <= max_idx)][
        ["date", "biz_day_index"]
    ].copy()

    if candidates.empty:
        return pd.DataFrame(columns=["as_of_date", "as_of_biz_day_index", "mae", "rmse", "wape"])

    if len(candidates) <= cfg.backtest_points:
        chosen = candidates
    else:
        picks = np.linspace(0, len(candidates) - 1, cfg.backtest_points).round().astype(int)
        chosen = candidates.iloc[picks].drop_duplicates(subset=["biz_day_index"])

    rows: list[dict] = []
    for r in chosen.itertuples(index=False):
        res = single_asof_forecast(con, cal, str(r.date), cfg)
        m = res["metrics"]
        rows.append(
            {
                "as_of_date": r.date,
                "as_of_biz_day_index": int(r.biz_day_index),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "wape": m["wape"],
            }
        )
    return pd.DataFrame(rows).sort_values("as_of_biz_day_index")
