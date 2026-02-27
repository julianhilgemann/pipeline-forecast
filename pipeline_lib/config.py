from __future__ import annotations

from dataclasses import dataclass


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
