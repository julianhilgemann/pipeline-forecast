from __future__ import annotations

import numpy as np
import pandas as pd


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


def forecast_business_day_range(cal: pd.DataFrame, as_of_idx: int, horizon: int) -> pd.DataFrame:
    fut = cal[
        (cal["is_business_day"])
        & (cal["biz_day_index"] > as_of_idx)
        & (cal["biz_day_index"] <= as_of_idx + horizon)
    ].copy()
    fut = fut.sort_values("biz_day_index")
    fut["day_offset"] = np.arange(1, len(fut) + 1, dtype=int)
    return fut[["date", "biz_day_index", "dow", "is_holiday", "is_business_day", "day_offset"]]


def default_asof_date(cal: pd.DataFrame, horizon: int) -> str:
    biz = cal[cal["is_business_day"]].sort_values("biz_day_index")
    target_idx = int(biz["biz_day_index"].max()) - horizon - 5
    row = biz[biz["biz_day_index"] == target_idx]
    if row.empty:
        row = biz.iloc[[max(0, len(biz) - horizon - 6)]]
    return str(row.iloc[0]["date"])
