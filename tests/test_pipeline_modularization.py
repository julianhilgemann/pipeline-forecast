from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

from pipeline_lib import Config, run_pipeline
from pipeline_lib.calendar import build_calendar, us_holidays_2025_2026
from pipeline_lib.overview.io import load_inputs
from pipeline_lib.overview.transforms import smooth_matrix, smooth_series


class PipelineModularizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out_dir = Path(cls.tmp.name)
        cfg = Config(
            start_date="2025-01-01",
            end_date="2025-05-31",
            seed=7,
            training_window_bdays=30,
            horizon_bdays=15,
            max_age_bdays=45,
            backtest_points=6,
            output_dir=str(cls.out_dir),
            db_file="test_pipeline.duckdb",
        )
        cls.run_result = run_pipeline(cfg)
        cls.asof_tag = cls.run_result["asof_tag"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_calendar_business_day_index_rules(self) -> None:
        cal = build_calendar("2025-01-01", "2025-01-10", us_holidays_2025_2026())
        idx = cal["biz_day_index"].to_numpy(dtype=int)
        is_bd = cal["is_business_day"].to_numpy(dtype=bool)

        self.assertEqual(idx[0], -1)  # 2025-01-01 holiday
        for i in range(1, len(idx)):
            if is_bd[i]:
                self.assertEqual(idx[i], idx[i - 1] + 1)
            else:
                self.assertEqual(idx[i], idx[i - 1])

    def test_pipeline_outputs_are_written(self) -> None:
        expected = [
            "daily_arrivals.csv",
            "outcomes_by_close_date.csv",
            "rolling_backtest_metrics.csv",
            f"active_stock_by_age_{self.asof_tag}.csv",
            f"win_kernel_{self.asof_tag}.csv",
            f"forecast_arrivals_by_day_{self.asof_tag}.csv",
            f"predicted_wins_by_day_{self.asof_tag}.csv",
            f"age_error_{self.asof_tag}.csv",
            f"stock_surface_{self.asof_tag}.csv",
            f"actual_stock_surface_{self.asof_tag}.csv",
            f"stock_error_surface_{self.asof_tag}.csv",
            "test_pipeline.duckdb",
        ]
        for name in expected:
            self.assertTrue((self.out_dir / name).exists(), msg=f"missing output: {name}")

        has_plot = any((self.out_dir / "figures").glob(f"*asof_{self.asof_tag}.*"))
        self.assertTrue(has_plot)

    def test_predicted_total_is_component_sum(self) -> None:
        pred = pd.read_csv(self.out_dir / f"predicted_wins_by_day_{self.asof_tag}.csv")
        lhs = pred["expected_wins_total"].to_numpy(dtype=float)
        rhs = (
            pred["expected_wins_stock"].to_numpy(dtype=float)
            + pred["expected_wins_arrivals"].to_numpy(dtype=float)
        )
        np.testing.assert_allclose(lhs, rhs, rtol=0.0, atol=1e-9)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib not installed")
    def test_overview_figure_generation_still_works(self) -> None:
        from pipeline_lib.overview.figure import create_figure

        data = load_inputs(self.out_dir, self.asof_tag)
        fig_path = self.out_dir / "figures" / f"overview_test_{self.asof_tag}.png"
        create_figure(
            asof_tag=self.asof_tag,
            kernel=data["kernel"],
            active_age=data["active_age"],
            arrivals_fc=data["arrivals_fc"],
            pred_by_day=data["pred_by_day"],
            stock_surface_df=data["stock_surface"],
            save_path=fig_path,
        )
        self.assertTrue(fig_path.exists())
        self.assertGreater(fig_path.stat().st_size, 0)

    def test_smoothing_helpers_preserve_shape(self) -> None:
        series = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        mat = np.arange(9, dtype=float).reshape(3, 3)
        self.assertEqual(smooth_series(series).shape, series.shape)
        self.assertEqual(smooth_matrix(mat).shape, mat.shape)


if __name__ == "__main__":
    unittest.main()
