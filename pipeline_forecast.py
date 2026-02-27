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
import sys

from pipeline_lib import Config, run_pipeline


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

    run = run_pipeline(cfg, as_of_date=args.as_of_date)
    result = run["result"]
    m = result["metrics"]

    print("=== Pipeline Forecast Prototype Complete ===")
    print(f"DuckDB: {run['db_path']}")
    print(
        f"Rows: dim_calendar={len(run['cal'])}, "
        f"fct_opportunity={len(run['opp'])}, "
        f"fct_event={len(run['ev'])}, "
        f"scd={len(run['scd'])}"
    )
    print(f"As-of date: {run['asof_tag']}")
    print(
        f"Daily metrics (total expected wins vs actual wins): "
        f"MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}, WAPE={m['wape']:.3f}"
    )
    print("Output CSVs:")
    for p in sorted(run["out_dir"].glob("*.csv")):
        print(f"  - {p}")
    print("Figures:")
    for fig_path in run["plot_paths"].values():
        print(f"  - {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
