#!/usr/bin/env python3
"""Streamlit interactive dashboard for Pipeline Forecast."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_lib.calendar import build_calendar, us_holidays_2025_2026
from pipeline_lib.config import Config
from pipeline_lib.forecasting import rolling_backtest, single_asof_forecast
from pipeline_lib.simulation import build_scd_status, simulate_events, simulate_opportunities
from pipeline_lib.warehouse import create_duckdb_tables

# ── Colour palette ─────────────────────────────────────────────────────────────
_BLUE   = "#4C9BE8"
_ORANGE = "#F28E2B"
_GREEN  = "#59A14F"
_RED    = "#E15759"


# ── Cached computations ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _sim(seed: int, start: str, end: str):
    """Run synthetic data generation; result is cached by (seed, start, end)."""
    rng = np.random.default_rng(seed)
    holidays = us_holidays_2025_2026()
    cal = build_calendar(start, end, holidays)
    switch_idx = int(cal["biz_day_index"].max() * 0.5)
    switch_date = str(
        cal[cal["is_business_day"] & (cal["biz_day_index"] >= switch_idx)]
        .sort_values("biz_day_index")
        .iloc[0]["date"]
    )
    opp = simulate_opportunities(cal, pd.Timestamp(switch_date), rng)
    ev  = simulate_events(opp, cal, switch_idx, rng)
    scd = build_scd_status(opp, ev)
    return cal, opp, ev, scd


@st.cache_data(show_spinner=False)
def _forecast(seed: int, start: str, end: str, as_of: str,
              tw: int, hz: int, ma: int) -> dict:
    """Run a single-as-of forecast; result is cached by all parameters."""
    cal, opp, ev, scd = _sim(seed, start, end)
    cfg = Config(
        start_date=start, end_date=end, seed=seed,
        training_window_bdays=tw, horizon_bdays=hz, max_age_bdays=ma,
    )
    con = create_duckdb_tables(Path(":memory:"), cal, opp, ev, scd)
    try:
        return single_asof_forecast(con, cal, as_of, cfg)
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def _backtest(seed: int, start: str, end: str,
              tw: int, hz: int, ma: int, bp: int) -> pd.DataFrame:
    """Run rolling backtest; result is cached by all parameters."""
    cal, opp, ev, scd = _sim(seed, start, end)
    cfg = Config(
        start_date=start, end_date=end, seed=seed,
        training_window_bdays=tw, horizon_bdays=hz,
        max_age_bdays=ma, backtest_points=bp,
    )
    con = create_duckdb_tables(Path(":memory:"), cal, opp, ev, scd)
    try:
        return rolling_backtest(con, cal, cfg)
    finally:
        con.close()


# ── Figure builders ────────────────────────────────────────────────────────────

def _kernel_fig(kernel: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=kernel["age_biz_days"], y=kernel["win_kernel_mass"],
        name="Win", mode="lines", fill="tozeroy",
        line=dict(color=_BLUE, width=2),
        fillcolor="rgba(76,155,232,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=kernel["age_biz_days"], y=kernel["loss_kernel_mass"],
        name="Loss", mode="lines",
        line=dict(color=_RED, width=1.5, dash="dot"),
    ))
    fig.update_layout(
        title="Win / Loss Kernel",
        xaxis_title="Age (business days)", yaxis_title="Probability mass",
        height=300, margin=dict(t=40, b=60, l=50, r=10),
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


def _stock_age_fig(active_age: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=active_age["age_biz_days"], y=active_age["active_count"],
        marker_color=_BLUE, marker_opacity=0.75,
    ))
    fig.update_layout(
        title="Active Pipeline by Age",
        xaxis_title="Age (business days)", yaxis_title="Opportunities",
        height=300, margin=dict(t=40, b=60, l=50, r=10),
    )
    return fig


def _heatmap_fig(surface: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=surface,
        colorscale="YlOrRd",
        colorbar=dict(title="Exp. wins", thickness=12),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Forecast day offset",
        yaxis_title="Age at as-of date (bdays)",
        height=380, margin=dict(t=40, b=60, l=60, r=10),
    )
    return fig


def _arrivals_fig(arrivals: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=pd.to_datetime(arrivals["date"]).dt.strftime("%Y-%m-%d"),
        y=arrivals["forecast_arrivals"],
        marker_color=_ORANGE, marker_opacity=0.8,
    ))
    fig.update_layout(
        title="Forecasted Daily Arrivals",
        xaxis=dict(title="Date", type="category"),
        yaxis_title="Arrivals",
        height=260, margin=dict(t=40, b=60, l=50, r=10),
    )
    return fig


def _daily_wins_fig(pred: pd.DataFrame) -> go.Figure:
    dates  = pd.to_datetime(pred["date"]).dt.strftime("%Y-%m-%d")
    total  = pred["expected_wins_total"].to_numpy(dtype=float)
    smooth = pd.Series(total).rolling(5, min_periods=1, center=True).mean().to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=pred["expected_wins_stock"],
        name="From stock", marker_color=_BLUE, marker_opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=dates, y=pred["expected_wins_arrivals"],
        name="From arrivals", marker_color=_ORANGE, marker_opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=smooth,
        name="Smoothed total", mode="lines",
        line=dict(color="black", width=2),
    ))
    if pred["actual_wins"].sum() > 0:
        fig.add_trace(go.Scatter(
            x=dates, y=pred["actual_wins"],
            name="Actual wins", mode="markers+lines",
            line=dict(color=_GREEN, width=1.5, dash="dash"),
            marker=dict(size=5),
        ))
    fig.update_layout(
        barmode="stack",
        title="Daily Expected Conversions",
        xaxis=dict(title="Date", type="category"),
        yaxis_title="Expected wins",
        height=400, margin=dict(t=40, b=80, l=50, r=10),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def _age_attr_fig(age_err: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=age_err["age_biz_days"],
        y=age_err["predicted_wins_from_stock"],
        name="Predicted", marker_color=_BLUE, opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=age_err["age_biz_days"],
        y=age_err["actual_wins_from_stock_cohort"],
        name="Actual", marker_color=_GREEN, opacity=0.75,
    ))
    fig.update_layout(
        barmode="group",
        title="Predicted vs Actual Wins from Active Stock — by Age Cohort",
        xaxis_title="Age at as-of date (bdays)",
        yaxis_title="Total wins over horizon",
        height=380, margin=dict(t=40, b=80, l=50, r=10),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def _backtest_fig(bt: pd.DataFrame) -> go.Figure:
    x = pd.to_datetime(bt["as_of_date"]).dt.strftime("%Y-%m-%d")
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["MAE", "RMSE", "WAPE"],
        vertical_spacing=0.1,
    )
    for row, (col, color) in enumerate(
        [("mae", _BLUE), ("rmse", _ORANGE), ("wape", _RED)], start=1
    ):
        fig.add_trace(
            go.Scatter(
                x=x, y=bt[col], mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=5),
            ),
            row=row, col=1,
        )
    fig.update_layout(
        title="Rolling Backtest — Forecast Accuracy Over Time",
        height=600, showlegend=False,
        margin=dict(t=60, b=60, l=60, r=10),
    )
    fig.update_xaxes(type="category")
    fig.update_xaxes(title_text="As-of date", row=3, col=1)
    return fig


# ── App ────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Pipeline Forecast", layout="wide")
    st.title("Pipeline Forecast Dashboard")

    # ── Sidebar ──
    with st.sidebar:
        st.header("Simulation")
        seed  = int(st.number_input("Random seed", value=42, min_value=0, step=1))
        start = st.text_input("Start date", value="2025-05-01")
        end   = st.text_input("End date",   value="2026-01-01")

        st.header("Forecast Parameters")
        tw = st.slider("Training window (bdays)",  30, 120,  90, 5)
        hz = st.slider("Forecast horizon (bdays)",  10,  60,  30, 5)
        ma = st.slider("Max age (bdays)",           30, 150,  30, 10)

        # Derive valid as-of date range from the simulation calendar
        try:
            cal, _, _, _ = _sim(seed, start, end)
        except Exception as exc:
            st.error(f"Simulation error: {exc}")
            return

        biz     = cal[cal["is_business_day"]]
        min_idx = tw + 5
        max_idx = int(biz["biz_day_index"].max()) - hz
        valid   = biz[
            (biz["biz_day_index"] >= min_idx) & (biz["biz_day_index"] <= max_idx)
        ]["date"]

        if valid.empty:
            st.error("No valid as-of dates. Reduce training window or horizon.")
            return

        valid_strs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in valid.tolist()]
        as_of = st.select_slider(
            "As-of date", options=valid_strs, value=valid_strs[-1]
        )

        st.divider()
        st.subheader("Backtest")
        bp     = st.slider("Backtest points", 5, 30, 10, 5)
        run_bt = st.button("Run Backtest", use_container_width=True)

    # ── Run forecast ──
    with st.spinner("Computing forecast…"):
        try:
            res = _forecast(seed, start, end, as_of, tw, hz, ma)
        except Exception as exc:
            st.error(f"Forecast error: {exc}")
            return

    m            = res["metrics"]
    active_count = int(res["active_age"]["active_count"].sum())
    total_exp    = float(res["pred_by_day"]["expected_wins_total"].sum())
    wape_label   = f"{m['wape']:.1%}" if np.isfinite(m["wape"]) else "N/A"

    # ── KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Pipeline",          f"{active_count:,}")
    c2.metric(f"Expected Wins ({hz}bd)",  f"{total_exp:.1f}")
    c3.metric("MAE",                      f"{m['mae']:.3f}")
    c4.metric("WAPE",                     wape_label)

    # ── Tabs ──
    t_fc, t_bt, t_age = st.tabs(["Forecast", "Backtest", "Age Attribution"])

    with t_fc:
        col_l, col_r = st.columns([1, 1.4])
        with col_l:
            st.plotly_chart(_kernel_fig(res["kernel"]),         use_container_width=True)
            st.plotly_chart(_stock_age_fig(res["active_age"]), use_container_width=True)
        with col_r:
            st.plotly_chart(
                _heatmap_fig(res["stock_surface"],
                             "Expected Wins from Active Stock (Age × Day)"),
                use_container_width=True,
            )
            st.plotly_chart(_arrivals_fig(res["arrivals_fc"]), use_container_width=True)
        st.plotly_chart(_daily_wins_fig(res["pred_by_day"]), use_container_width=True)

    with t_bt:
        if run_bt:
            st.session_state["_bt_triggered"] = True
        if st.session_state.get("_bt_triggered"):
            with st.spinner("Running backtest (this may take a moment)…"):
                bt = _backtest(seed, start, end, tw, hz, ma, bp)
            st.plotly_chart(_backtest_fig(bt), use_container_width=True)
        else:
            st.info("Click **Run Backtest** in the sidebar to compute rolling accuracy metrics.")

    with t_age:
        st.plotly_chart(_age_attr_fig(res["age_error"]), use_container_width=True)


if __name__ == "__main__":
    main()
