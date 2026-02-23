#!/usr/bin/env python3
"""
Business-day sales pipeline forecasting prototype with:
- competing outcomes (won/lost/censored)
- inflow forecasting
- age-based stock projection
- single as-of heatmap surface
- rolling backtest metrics
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


@dataclass
class Config:
    start_date: str = "2025-01-01"
    end_date: str = "2026-01-31"
    seed: int = 42
    training_window_bdays: int = 60
    horizon_bdays: int = 30
    max_age_bdays: int = 90
    backtest_points: int = 20
    output_dir: str = "outputs"
    db_file: str = "pipeline_forecast.duckdb"


def us_holidays_2025_2026() -> set[pd.Timestamp]:
    holiday_str = [
        "2025-01-01",
        "2025-01-20",
        "2025-02-17",
        "2025-05-26",
        "2025-07-04",
        "2025-09-01",
        "2025-11-27",
        "2025-12-25",
        "2026-01-01",
        "2026-01-19",
    ]
    return {pd.Timestamp(d) for d in holiday_str}


def build_calendar(start_date: str, end_date: str, holidays: set[pd.Timestamp]) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["dow"] = cal["date"].dt.dayofweek.astype(int)
    cal["is_weekend"] = cal["dow"] >= 5
    cal["is_holiday"] = cal["date"].isin(holidays)
    cal["is_business_day"] = ~(cal["is_weekend"] | cal["is_holiday"])
    cal["year"] = cal["date"].dt.year.astype(int)
    cal["month"] = cal["date"].dt.month.astype(int)
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)

    biz_idx: list[int] = []
    running = -1
    for is_bd in cal["is_business_day"].tolist():
        if bool(is_bd):
            running += 1
        biz_idx.append(running)
    cal["biz_day_index"] = np.array(biz_idx, dtype=int)
    cal["date"] = cal["date"].dt.date
    return cal


def draw_deal_size(segment: str, rng: np.random.Generator) -> float:
    params = {
        "SMB": (8.9, 0.55),
        "MidMarket": (10.1, 0.60),
        "Enterprise": (11.2, 0.70),
    }
    mu, sigma = params[segment]
    return float(np.exp(rng.normal(mu, sigma)))


def simulate_opportunities(
    cal: pd.DataFrame, regime_switch_date: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    seg_values = np.array(["SMB", "MidMarket", "Enterprise"])
    seg_probs = np.array([0.60, 0.30, 0.10])
    ch_values = np.array(["Inbound", "Outbound", "Partner"])
    ch_probs = np.array([0.48, 0.37, 0.15])
    dow_mult = {0: 1.25, 1: 1.15, 2: 1.00, 3: 0.95, 4: 0.80, 5: 0.45, 6: 0.35}

    rows: list[dict] = []
    opp_id = 1
    for r in cal.itertuples(index=False):
        date_ts = pd.Timestamp(r.date)
        base = 12.0 if r.is_business_day else 2.2
        lam = base * dow_mult[int(r.dow)]
        if r.is_holiday:
            lam *= 0.25
        if date_ts >= regime_switch_date:
            lam *= 1.20
        lam = max(lam, 0.05)
        n = int(rng.poisson(lam=lam))
        if n == 0:
            continue

        segs = rng.choice(seg_values, size=n, p=seg_probs)
        chs = rng.choice(ch_values, size=n, p=ch_probs)
        for i in range(n):
            segment = str(segs[i])
            rows.append(
                {
                    "opp_id": opp_id,
                    "created_date": r.date,
                    "created_biz_day_index": int(r.biz_day_index),
                    "segment": segment,
                    "channel": str(chs[i]),
                    "deal_size": draw_deal_size(segment, rng),
                }
            )
            opp_id += 1
    opp = pd.DataFrame(rows)
    opp["deal_size"] = opp["deal_size"].round(2)
    return opp


def _clip_prob(v: float, low: float = 0.01, high: float = 0.99) -> float:
    return float(min(high, max(low, v)))


def simulate_events(
    opp: pd.DataFrame, cal: pd.DataFrame, regime_switch_biz_idx: int, rng: np.random.Generator
) -> pd.DataFrame:
    biz_map = (
        cal[cal["is_business_day"]]
        .drop_duplicates("biz_day_index")
        .set_index("biz_day_index")["date"]
        .to_dict()
    )
    max_biz_idx = int(cal["biz_day_index"].max())

    seg_close_adj = {"SMB": 0.03, "MidMarket": 0.00, "Enterprise": -0.06}
    ch_close_adj = {"Inbound": 0.04, "Outbound": -0.03, "Partner": 0.01}
    seg_win_adj = {"SMB": -0.02, "MidMarket": 0.03, "Enterprise": 0.08}
    ch_win_adj = {"Inbound": 0.06, "Outbound": -0.04, "Partner": 0.02}
    seg_age_shift = {"SMB": 0.0, "MidMarket": 3.0, "Enterprise": 8.0}
    ch_age_shift = {"Inbound": -1.0, "Outbound": 2.0, "Partner": 1.0}

    rows: list[dict] = []
    for r in opp.itertuples(index=False):
        created_idx = int(r.created_biz_day_index)
        post_regime = created_idx >= regime_switch_biz_idx

        p_close = 0.86 + seg_close_adj[r.segment] + ch_close_adj[r.channel]
        if post_regime:
            p_close -= 0.08
        p_close = _clip_prob(p_close, 0.30, 0.98)

        if rng.random() > p_close:
            rows.append(
                {
                    "opp_id": int(r.opp_id),
                    "outcome": "censored",
                    "closed_date": None,
                    "closed_biz_day_index": pd.NA,
                    "time_to_close_biz_days": pd.NA,
                }
            )
            continue

        raw_ttc = rng.gamma(shape=2.4, scale=6.0)
        ttc = int(round(raw_ttc + seg_age_shift[r.segment] + ch_age_shift[r.channel]))
        ttc = max(1, ttc)
        if post_regime:
            ttc = int(round(ttc * 1.35 + rng.integers(1, 5)))
            ttc = max(1, ttc)

        closed_idx = created_idx + ttc
        if closed_idx > max_biz_idx or closed_idx not in biz_map:
            rows.append(
                {
                    "opp_id": int(r.opp_id),
                    "outcome": "censored",
                    "closed_date": None,
                    "closed_biz_day_index": pd.NA,
                    "time_to_close_biz_days": pd.NA,
                }
            )
            continue

        p_win = 0.29 + seg_win_adj[r.segment] + ch_win_adj[r.channel] + (0.0016 * ttc)
        if post_regime:
            p_win -= 0.12
        p_win = _clip_prob(p_win, 0.05, 0.90)

        outcome = "won" if rng.random() < p_win else "lost"
        rows.append(
            {
                "opp_id": int(r.opp_id),
                "outcome": outcome,
                "closed_date": biz_map[closed_idx],
                "closed_biz_day_index": int(closed_idx),
                "time_to_close_biz_days": int(ttc),
            }
        )

    ev = pd.DataFrame(rows)
    ev["closed_biz_day_index"] = ev["closed_biz_day_index"].astype("Int64")
    ev["time_to_close_biz_days"] = ev["time_to_close_biz_days"].astype("Int64")
    return ev


def build_scd_status(opp: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    merged = opp.merge(ev, on="opp_id", how="left")
    rows: list[dict] = []
    for r in merged.itertuples(index=False):
        rows.append(
            {
                "opp_id": int(r.opp_id),
                "valid_from_date": r.created_date,
                "valid_to_date": r.closed_date if r.outcome in ("won", "lost") else None,
                "status": "active",
            }
        )
        if r.outcome in ("won", "lost"):
            rows.append(
                {
                    "opp_id": int(r.opp_id),
                    "valid_from_date": r.closed_date,
                    "valid_to_date": None,
                    "status": r.outcome,
                }
            )
    return pd.DataFrame(rows)


def create_duckdb_tables(
    db_path: Path, cal: pd.DataFrame, opp: pd.DataFrame, ev: pd.DataFrame, scd: pd.DataFrame
) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.register("cal_df", cal)
    con.register("opp_df", opp)
    con.register("ev_df", ev)
    con.register("scd_df", scd)

    con.execute(
        """
        CREATE OR REPLACE TABLE dim_calendar AS
        SELECT
            CAST(date AS DATE) AS date,
            CAST(is_weekend AS BOOLEAN) AS is_weekend,
            CAST(is_holiday AS BOOLEAN) AS is_holiday,
            CAST(is_business_day AS BOOLEAN) AS is_business_day,
            CAST(biz_day_index AS INTEGER) AS biz_day_index,
            CAST(dow AS INTEGER) AS dow,
            CAST(year AS INTEGER) AS year,
            CAST(month AS INTEGER) AS month,
            CAST(week AS INTEGER) AS week
        FROM cal_df
        ORDER BY date
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE fct_opportunity AS
        SELECT
            CAST(opp_id AS BIGINT) AS opp_id,
            CAST(created_date AS DATE) AS created_date,
            CAST(created_biz_day_index AS INTEGER) AS created_biz_day_index,
            CAST(segment AS VARCHAR) AS segment,
            CAST(channel AS VARCHAR) AS channel,
            CAST(deal_size AS DOUBLE) AS deal_size
        FROM opp_df
        ORDER BY opp_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE fct_opportunity_event AS
        SELECT
            CAST(opp_id AS BIGINT) AS opp_id,
            CAST(outcome AS VARCHAR) AS outcome,
            CAST(closed_date AS DATE) AS closed_date,
            CAST(closed_biz_day_index AS INTEGER) AS closed_biz_day_index,
            CAST(time_to_close_biz_days AS INTEGER) AS time_to_close_biz_days
        FROM ev_df
        ORDER BY opp_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE scd_opportunity_status AS
        SELECT
            CAST(opp_id AS BIGINT) AS opp_id,
            CAST(valid_from_date AS DATE) AS valid_from_date,
            CAST(valid_to_date AS DATE) AS valid_to_date,
            CAST(status AS VARCHAR) AS status
        FROM scd_df
        ORDER BY opp_id, valid_from_date, status
        """
    )
    return con


def sql_daily_arrivals(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            created_date,
            COUNT(*) AS arrivals
        FROM fct_opportunity
        GROUP BY created_date
        ORDER BY created_date
        """
    ).df()


def sql_outcomes_by_close_date(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            closed_date,
            outcome,
            COUNT(*) AS n_outcomes
        FROM fct_opportunity_event
        WHERE closed_date IS NOT NULL
        GROUP BY closed_date, outcome
        ORDER BY closed_date, outcome
        """
    ).df()


def active_snapshot_asof(
    con: duckdb.DuckDBPyConnection, as_of_date: str, as_of_idx: int, max_age: int
) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            LEAST(?, GREATEST(0, ? - o.created_biz_day_index)) AS age_biz_days,
            COUNT(*) AS active_count
        FROM scd_opportunity_status s
        JOIN fct_opportunity o USING (opp_id)
        WHERE s.valid_from_date <= CAST(? AS DATE)
          AND (s.valid_to_date > CAST(? AS DATE) OR s.valid_to_date IS NULL)
          AND s.status = 'active'
        GROUP BY 1
        ORDER BY 1
        """,
        [max_age, as_of_idx, as_of_date, as_of_date],
    ).df()


def active_opportunities_asof(
    con: duckdb.DuckDBPyConnection, as_of_date: str, as_of_idx: int, max_age: int
) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            o.opp_id,
            LEAST(?, GREATEST(0, ? - o.created_biz_day_index)) AS age_biz_days
        FROM scd_opportunity_status s
        JOIN fct_opportunity o USING (opp_id)
        WHERE s.valid_from_date <= CAST(? AS DATE)
          AND (s.valid_to_date > CAST(? AS DATE) OR s.valid_to_date IS NULL)
          AND s.status = 'active'
        """,
        [max_age, as_of_idx, as_of_date, as_of_date],
    ).df()


def training_window_extract(
    con: duckdb.DuckDBPyConnection, train_start_idx: int, train_end_idx: int
) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            o.opp_id,
            o.created_biz_day_index,
            e.outcome,
            e.closed_biz_day_index,
            e.time_to_close_biz_days
        FROM fct_opportunity o
        JOIN fct_opportunity_event e USING (opp_id)
        WHERE o.created_biz_day_index <= ?
          AND (e.closed_biz_day_index IS NULL OR e.closed_biz_day_index >= ?)
        """,
        [train_end_idx, train_start_idx],
    ).df()


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


def forecast_business_day_range(
    cal: pd.DataFrame, as_of_idx: int, horizon: int
) -> pd.DataFrame:
    fut = cal[
        (cal["is_business_day"])
        & (cal["biz_day_index"] > as_of_idx)
        & (cal["biz_day_index"] <= as_of_idx + horizon)
    ].copy()
    fut = fut.sort_values("biz_day_index")
    fut["day_offset"] = np.arange(1, len(fut) + 1, dtype=int)
    return fut[["date", "biz_day_index", "dow", "is_holiday", "is_business_day", "day_offset"]]


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


def actual_wins_by_day(
    con: duckdb.DuckDBPyConnection, as_of_idx: int, horizon: int
) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            closed_biz_day_index AS biz_day_index,
            COUNT(*) AS actual_wins
        FROM fct_opportunity_event
        WHERE outcome = 'won'
          AND closed_biz_day_index BETWEEN ? AND ?
        GROUP BY closed_biz_day_index
        ORDER BY closed_biz_day_index
        """,
        [as_of_idx + 1, as_of_idx + horizon],
    ).df()


def daily_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum(np.abs(actual)))
    wape = float(np.sum(np.abs(err)) / denom) if denom > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "wape": wape}


def cohort_actual_surface(
    con: duckdb.DuckDBPyConnection,
    active_opp: pd.DataFrame,
    as_of_idx: int,
    horizon: int,
    max_age: int,
) -> np.ndarray:
    surface = np.zeros((max_age + 1, horizon), dtype=float)
    if active_opp.empty:
        return surface

    con.register("active_ids_df", active_opp[["opp_id", "age_biz_days"]])
    actual = con.execute(
        """
        SELECT
            a.age_biz_days,
            e.closed_biz_day_index - ? AS day_offset,
            COUNT(*) AS n_wins
        FROM active_ids_df a
        JOIN fct_opportunity_event e USING (opp_id)
        WHERE e.outcome = 'won'
          AND e.closed_biz_day_index BETWEEN ? AND ?
        GROUP BY 1, 2
        ORDER BY 1, 2
        """,
        [as_of_idx, as_of_idx + 1, as_of_idx + horizon],
    ).df()
    for r in actual.itertuples(index=False):
        age = int(r.age_biz_days)
        day_offset = int(r.day_offset)
        if 0 <= age <= max_age and 1 <= day_offset <= horizon:
            surface[age, day_offset - 1] = float(r.n_wins)
    return surface


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


def default_asof_date(cal: pd.DataFrame, horizon: int) -> str:
    biz = cal[cal["is_business_day"]].sort_values("biz_day_index")
    target_idx = int(biz["biz_day_index"].max()) - horizon - 5
    row = biz[biz["biz_day_index"] == target_idx]
    if row.empty:
        row = biz.iloc[[max(0, len(biz) - horizon - 6)]]
    return str(row.iloc[0]["date"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Business-day sales pipeline forecast prototype")
    p.add_argument("--start-date", default=Config.start_date)
    p.add_argument("--end-date", default=Config.end_date)
    p.add_argument("--as-of-date", default=None, help="Business-day as-of date (YYYY-MM-DD)")
    p.add_argument("--training-window", type=int, default=Config.training_window_bdays)
    p.add_argument("--horizon", type=int, default=Config.horizon_bdays)
    p.add_argument("--max-age", type=int, default=Config.max_age_bdays)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--backtest-points", type=int, default=Config.backtest_points)
    p.add_argument("--output-dir", default=Config.output_dir)
    p.add_argument("--db-file", default=Config.db_file)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed,
        training_window_bdays=args.training_window,
        horizon_bdays=args.horizon,
        max_age_bdays=args.max_age,
        backtest_points=args.backtest_points,
        output_dir=args.output_dir,
        db_file=args.db_file,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / cfg.db_file

    rng = np.random.default_rng(cfg.seed)
    holidays = us_holidays_2025_2026()
    cal = build_calendar(cfg.start_date, cfg.end_date, holidays)
    regime_switch_idx = int(cal["biz_day_index"].max() * 0.5)
    regime_switch_date = (
        cal[(cal["is_business_day"]) & (cal["biz_day_index"] >= regime_switch_idx)]
        .sort_values("biz_day_index")
        .iloc[0]["date"]
    )
    opp = simulate_opportunities(cal, pd.Timestamp(regime_switch_date), rng)
    ev = simulate_events(opp, cal, regime_switch_idx, rng)
    scd = build_scd_status(opp, ev)

    con = create_duckdb_tables(db_path, cal, opp, ev, scd)
    try:
        daily_arrivals = sql_daily_arrivals(con)
        outcomes_by_date = sql_outcomes_by_close_date(con)
        daily_arrivals.to_csv(out_dir / "daily_arrivals.csv", index=False)
        outcomes_by_date.to_csv(out_dir / "outcomes_by_close_date.csv", index=False)

        as_of_date = args.as_of_date if args.as_of_date else default_asof_date(cal, cfg.horizon_bdays)
        result = single_asof_forecast(con, cal, as_of_date, cfg)

        asof_tag = str(result["as_of_date"])
        result["active_age"].to_csv(out_dir / f"active_stock_by_age_{asof_tag}.csv", index=False)
        result["kernel"].to_csv(out_dir / f"win_kernel_{asof_tag}.csv", index=False)
        result["arrivals_fc"].to_csv(out_dir / f"forecast_arrivals_by_day_{asof_tag}.csv", index=False)
        result["pred_by_day"].to_csv(out_dir / f"predicted_wins_by_day_{asof_tag}.csv", index=False)
        result["age_error"].to_csv(out_dir / f"age_error_{asof_tag}.csv", index=False)

        stock_surface = result["stock_surface"]
        actual_stock_surface = result["actual_stock_surface"]
        stock_surface_df = pd.DataFrame(stock_surface)
        stock_surface_df.insert(0, "age_biz_days", np.arange(cfg.max_age_bdays + 1, dtype=int))
        stock_surface_df.columns = ["age_biz_days"] + [
            f"offset_{i}" for i in range(1, cfg.horizon_bdays + 1)
        ]
        stock_surface_df.to_csv(out_dir / f"stock_surface_{asof_tag}.csv", index=False)

        actual_surface_df = pd.DataFrame(actual_stock_surface)
        actual_surface_df.insert(0, "age_biz_days", np.arange(cfg.max_age_bdays + 1, dtype=int))
        actual_surface_df.columns = ["age_biz_days"] + [
            f"offset_{i}" for i in range(1, cfg.horizon_bdays + 1)
        ]
        actual_surface_df.to_csv(out_dir / f"actual_stock_surface_{asof_tag}.csv", index=False)

        error_surface = stock_surface - actual_stock_surface
        error_surface_df = pd.DataFrame(error_surface)
        error_surface_df.insert(0, "age_biz_days", np.arange(cfg.max_age_bdays + 1, dtype=int))
        error_surface_df.columns = ["age_biz_days"] + [
            f"offset_{i}" for i in range(1, cfg.horizon_bdays + 1)
        ]
        error_surface_df.to_csv(out_dir / f"stock_error_surface_{asof_tag}.csv", index=False)

        plot_paths = render_plots(
            out_dir, asof_tag, result["stock_surface"], result["pred_by_day"]
        )

        backtest = rolling_backtest(con, cal, cfg)
        backtest.to_csv(out_dir / "rolling_backtest_metrics.csv", index=False)

        m = result["metrics"]
        print("=== Pipeline Forecast Prototype Complete ===")
        print(f"DuckDB: {db_path}")
        print(f"Rows: dim_calendar={len(cal)}, fct_opportunity={len(opp)}, fct_event={len(ev)}, scd={len(scd)}")
        print(f"As-of date: {asof_tag}")
        print(
            f"Daily metrics (total expected wins vs actual wins): "
            f"MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}, WAPE={m['wape']:.3f}"
        )
        print("Output CSVs:")
        for p in sorted(out_dir.glob("*.csv")):
            print(f"  - {p}")
        print("Figures:")
        for _, fig_path in plot_paths.items():
            print(f"  - {fig_path}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
