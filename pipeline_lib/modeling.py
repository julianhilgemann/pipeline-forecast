from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd


def estimate_kernel_from_training(
    train_df: pd.DataFrame,
    train_start_idx: int,
    train_end_idx: int,
    max_age: int,
    smoothing: float = 0.25,
) -> pd.DataFrame:
    exposures = np.zeros(max_age + 1, dtype=float)
    wins = np.zeros(max_age + 1, dtype=float)
    losses = np.zeros(max_age + 1, dtype=float)

    for r in train_df.itertuples(index=False):
        created = int(r.created_biz_day_index)
        close_idx = None if pd.isna(r.closed_biz_day_index) else int(r.closed_biz_day_index)
        close_age = None if pd.isna(r.time_to_close_biz_days) else int(r.time_to_close_biz_days)
        outcome = r.outcome

        if created > train_end_idx:
            continue
        if close_idx is not None and close_idx < train_start_idx:
            continue

        start_age = max(0, train_start_idx - created)
        if close_idx is not None and close_idx <= train_end_idx and close_age is not None:
            end_age = min(close_age, max_age)
            event_age = close_age
            has_event = outcome in ("won", "lost")
        else:
            end_age = min(train_end_idx - created, max_age)
            event_age = None
            has_event = False

        if end_age < start_age:
            continue

        exposures[start_age : end_age + 1] += 1.0
        if has_event and event_age is not None and 0 <= event_age <= max_age:
            if outcome == "won":
                wins[event_age] += 1.0
            elif outcome == "lost":
                losses[event_age] += 1.0

    win_hazard = np.zeros(max_age + 1, dtype=float)
    loss_hazard = np.zeros(max_age + 1, dtype=float)
    at_risk = exposures > 0
    denom = exposures[at_risk] + (2.0 * smoothing)
    win_hazard[at_risk] = (wins[at_risk] + smoothing) / denom
    loss_hazard[at_risk] = (losses[at_risk] + smoothing) / denom

    total_h = win_hazard + loss_hazard
    too_big = total_h > 0.999
    if np.any(too_big):
        scale = 0.999 / total_h[too_big]
        win_hazard[too_big] = win_hazard[too_big] * scale
        loss_hazard[too_big] = loss_hazard[too_big] * scale

    survival_to_age = np.zeros(max_age + 1, dtype=float)
    win_mass = np.zeros(max_age + 1, dtype=float)
    loss_mass = np.zeros(max_age + 1, dtype=float)
    surv = 1.0
    for age in range(max_age + 1):
        survival_to_age[age] = surv
        win_mass[age] = surv * win_hazard[age]
        loss_mass[age] = surv * loss_hazard[age]
        surv = surv * max(0.0, 1.0 - win_hazard[age] - loss_hazard[age])

    return pd.DataFrame(
        {
            "age_biz_days": np.arange(max_age + 1, dtype=int),
            "at_risk_count": exposures,
            "wins_at_age": wins,
            "losses_at_age": losses,
            "win_hazard": win_hazard,
            "loss_hazard": loss_hazard,
            "survival_to_age": survival_to_age,
            "win_kernel_mass": win_mass,
            "loss_kernel_mass": loss_mass,
        }
    )


def project_wins_from_active_stock(
    active_by_age: pd.DataFrame, kernel_df: pd.DataFrame, horizon: int, max_age: int
) -> tuple[pd.DataFrame, np.ndarray]:
    win_mass = kernel_df["win_kernel_mass"].to_numpy(dtype=float)
    survival = kernel_df["survival_to_age"].to_numpy(dtype=float)
    surface = np.zeros((max_age + 1, horizon), dtype=float)
    totals = np.zeros(horizon, dtype=float)

    for r in active_by_age.itertuples(index=False):
        age = int(r.age_biz_days)
        n = float(r.active_count)
        if n <= 0 or age < 0 or age > max_age:
            continue

        denom = survival[age]
        if denom <= 1e-12:
            continue

        for day_idx in range(horizon):
            abs_age = age + day_idx + 1
            if abs_age > max_age:
                break
            cond_prob = win_mass[abs_age] / denom
            expected = n * cond_prob
            surface[age, day_idx] += expected
            totals[day_idx] += expected

    return (
        pd.DataFrame(
            {"day_offset": np.arange(1, horizon + 1, dtype=int), "expected_wins_stock": totals}
        ),
        surface,
    )


def seasonal_naive_arrival_forecast(
    con: duckdb.DuckDBPyConnection, as_of_date: str, future_bdays: pd.DataFrame
) -> pd.DataFrame:
    hist = con.execute(
        """
        WITH arrivals AS (
            SELECT created_date AS date, COUNT(*) AS arrivals
            FROM fct_opportunity
            GROUP BY created_date
        )
        SELECT
            c.date,
            c.dow,
            c.is_holiday,
            c.is_business_day,
            COALESCE(a.arrivals, 0) AS arrivals
        FROM dim_calendar c
        LEFT JOIN arrivals a ON c.date = a.date
        WHERE c.date <= CAST(? AS DATE)
        ORDER BY c.date
        """,
        [as_of_date],
    ).df()

    recent = hist.tail(180).copy()
    global_mean = float(recent["arrivals"].mean()) if len(recent) else 0.0
    biz_dow_mean = (
        recent[recent["is_business_day"]].groupby("dow", as_index=True)["arrivals"].mean().to_dict()
    )
    nonbiz_mean = float(recent[~recent["is_business_day"]]["arrivals"].mean())
    holiday_mean = float(recent[recent["is_holiday"]]["arrivals"].mean())

    if math.isnan(nonbiz_mean):
        nonbiz_mean = max(0.2, 0.2 * global_mean)
    if math.isnan(holiday_mean):
        holiday_mean = nonbiz_mean

    preds: list[float] = []
    for r in future_bdays.itertuples(index=False):
        if bool(r.is_holiday):
            pred = holiday_mean
        elif bool(r.is_business_day):
            pred = float(biz_dow_mean.get(int(r.dow), global_mean))
        else:
            pred = nonbiz_mean
        preds.append(max(0.05, pred))

    out = future_bdays[["date", "biz_day_index", "day_offset"]].copy()
    out["forecast_arrivals"] = np.array(preds, dtype=float)
    return out


def project_wins_from_arrivals(
    arrivals_forecast: pd.DataFrame, kernel_df: pd.DataFrame, horizon: int, max_age: int
) -> pd.DataFrame:
    win_mass = kernel_df["win_kernel_mass"].to_numpy(dtype=float)
    arrivals = arrivals_forecast.sort_values("day_offset")["forecast_arrivals"].to_numpy(dtype=float)
    totals = np.zeros(horizon, dtype=float)

    for create_idx in range(horizon):
        n_arr = arrivals[create_idx]
        if n_arr <= 0:
            continue
        for close_idx in range(create_idx, horizon):
            age = close_idx - create_idx
            if age > max_age:
                break
            totals[close_idx] += n_arr * win_mass[age]

    return pd.DataFrame(
        {"day_offset": np.arange(1, horizon + 1, dtype=int), "expected_wins_arrivals": totals}
    )


def daily_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum(np.abs(actual)))
    wape = float(np.sum(np.abs(err)) / denom) if denom > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "wape": wape}
