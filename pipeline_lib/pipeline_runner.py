from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import build_calendar, default_asof_date, us_holidays_2025_2026
from .config import Config
from .data_io import write_single_forecast_outputs
from .forecasting import rolling_backtest, single_asof_forecast
from .plotting import render_plots
from .simulation import build_scd_status, simulate_events, simulate_opportunities
from .warehouse import create_duckdb_tables, sql_daily_arrivals, sql_outcomes_by_close_date


def run_pipeline(cfg: Config, as_of_date: str | None = None) -> dict[str, object]:
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

        forecast_asof = as_of_date if as_of_date else default_asof_date(cal, cfg.horizon_bdays)
        result = single_asof_forecast(con, cal, forecast_asof, cfg)
        asof_tag = str(result["as_of_date"])
        write_single_forecast_outputs(out_dir, asof_tag, result, cfg)

        plot_paths = render_plots(
            out_dir,
            asof_tag,
            result["stock_surface"],
            result["pred_by_day"],
        )

        backtest = rolling_backtest(con, cal, cfg)
        backtest.to_csv(out_dir / "rolling_backtest_metrics.csv", index=False)

        return {
            "db_path": db_path,
            "cal": cal,
            "opp": opp,
            "ev": ev,
            "scd": scd,
            "asof_tag": asof_tag,
            "result": result,
            "plot_paths": plot_paths,
            "out_dir": out_dir,
        }
    finally:
        con.close()
