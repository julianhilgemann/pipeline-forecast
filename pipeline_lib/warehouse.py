from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


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


def actual_wins_by_day(con: duckdb.DuckDBPyConnection, as_of_idx: int, horizon: int) -> pd.DataFrame:
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
