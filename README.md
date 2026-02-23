# Pipeline Forecast

Business-day sales pipeline forecasting prototype with:
- competing outcomes (`won`, `lost`, `censored`)
- age-based conversion kernel estimation
- point-in-time (PIT) reconstruction from SCD2 status history
- expected wins projection from current stock + forecasted arrivals
- multi-panel visualization of the full forecast mechanics

## Objective

Model how a B2B opportunity pipeline converts over time (in business days), then forecast future daily conversions by combining:
1. conversions expected from the existing active stock
2. conversions expected from newly arriving opportunities

This repository focuses on a transparent, auditable forecasting framework that can evolve into an interactive planning dashboard.

## Core Data Architecture (DuckDB)

The warehouse is generated into DuckDB with four core tables:

### `dim_calendar`
- `date` (PK)
- weekend/holiday flags
- `is_business_day`
- `biz_day_index` (increments only on business days)
- day/time dimensions (`dow`, `year`, `month`, `week`)

Purpose:
- canonical business-day clock
- converts calendar dates into business-day age/offset units

### `fct_opportunity`
One row per opportunity:
- `opp_id` (PK)
- `created_date`
- `created_biz_day_index`
- `segment`, `channel`
- `deal_size`

Purpose:
- stock/inflow base table
- segmentation and revenue context

### `fct_opportunity_event`
One terminal event per opportunity (or censored):
- `opp_id`
- `outcome` (`won`, `lost`, `censored`)
- `closed_date`
- `closed_biz_day_index`
- `time_to_close_biz_days`

Purpose:
- competing-outcome event history
- age-at-close extraction in business days

### `scd_opportunity_status` (SCD2)
Status history with validity ranges:
- `opp_id`
- `valid_from_date` (inclusive)
- `valid_to_date` (exclusive, null = current)
- `status` (`active`, `won`, `lost`)

Purpose:
- robust PIT reconstruction of active pipeline as-of any date
- prevents look-ahead leakage in snapshot/backtest workflows

## SCD2 + PIT Reconstruction Logic

Each opportunity is modeled with:
1. `active` row from `created_date` to `closed_date` (or open-ended if censored)
2. terminal row (`won` or `lost`) from `closed_date` onward

PIT active snapshot filter:
- `valid_from_date <= as_of_date`
- `valid_to_date > as_of_date OR valid_to_date IS NULL`
- `status = 'active'`

Age at snapshot:
- `age_biz_days = as_of_biz_day_index - created_biz_day_index`

This is the foundation for unbiased as-of forecasting and walk-forward validation.

## Synthetic Data Generating Process

The generator intentionally includes realistic pipeline behaviors:
- business-day seasonality in arrivals
- weekend/holiday suppression
- segment/channel differences in close timing and win rate
- censoring (open deals)
- regime switch mid-period (timing/win-rate shift)

The regime change is designed to be visible in forecast surfaces and backtest metrics.

## Forecasting Mechanism

### 1) Active stock by age (PIT)
Reconstruct active opportunities as-of a business date and aggregate counts by age bucket.

### 2) Competing-risk kernel estimation
From a trailing training window:
- compute at-risk exposure by age
- count wins/losses by age
- estimate age-wise win/loss hazards
- transform hazards into win kernel mass by age

Interpretation:
- kernel mass at age `a` = probability that a new opportunity wins at age `a`

### 3) Wins from existing stock
For each current age bucket:
- condition kernel on survival to current age
- spread expected wins over future business-day horizon

Output:
- daily expected wins from stock
- age x day transition surface (heatmap matrix)

### 4) Arrival forecast
Forecast daily future opportunity inflow (seasonal-naive baseline in current implementation).

### 5) Wins from forecasted arrivals
Treat each forecast day as an arrival cohort at age 0 and convolve with kernel.

Output:
- daily expected wins from arrivals
- arrival-day x conversion-day transition surface

### 6) Combined expected conversions
- `expected_total = expected_from_stock + expected_from_arrivals`

## Evaluation Framework

Current evaluation includes:
- daily MAE / RMSE / WAPE
- rolling multi-as-of backtest
- age/cohort-attributable error surfaces (stock attribution)

This enables both:
- time-axis performance view
- age/cohort diagnostics

## Visualization Layer

`model_overview_figure.py` renders a structured multi-panel forecast architecture:
- smoothed positive decision kernel
- existing-pipeline convolution heatmap + aligned side distributions
- future-arrivals convolution heatmap + aligned side distributions
- stacked daily expected conversions with smoothed total

## Repository Structure

- [`pipeline_forecast.py`](/Users/admin/Desktop/pipeline-forecast/pipeline_forecast.py)
  End-to-end synthetic generation, DuckDB write, forecasting, evaluation, and exports.
- [`model_overview_figure.py`](/Users/admin/Desktop/pipeline-forecast/model_overview_figure.py)
  Multi-panel architecture figure from exported forecast tables.
- [`outputs/`](/Users/admin/Desktop/pipeline-forecast/outputs)
  Generated DuckDB, CSVs, and figures.

## Run

1. Generate data + forecast outputs:
```bash
python pipeline_forecast.py
```

2. Render architecture figure:
```bash
python model_overview_figure.py --output-dir outputs
```

## Roadmap / Next Steps

### Productization and UX
- Add main KPI cards (pipeline stock, projected wins, target gap, QoQ view)
- Wrap analysis in an interactive dashboard for business users
- Improve business narrative framing (quota coverage, shortfall risks, intervention levers)

### Validation and Model Risk
- Validation KPI suite (bias, calibration, drift, stability bands)
- Walk-forward backtesting and PIT reconstruction as first-class workflows
- Kernel stability metrics over time windows
- Kernel calibration diagnostics
- Residual decomposition to quantify error contributors:
  - stock vs arrivals
  - age buckets
  - horizon segments
  - segment/channel slices

### Uncertainty and Decision Support
- Add confidence intervals / uncertainty bands for expected conversions
- Add scenario overlays (upside/base/downside)
- Expose uncertainty perspectives for planning and governance

### Revenue Forecasting
- Add probability-weighted revenue column:
  - expected revenue = expected wins x segment-adjusted deal-size assumptions
- Support revenue-weighted objectives alongside volume KPIs

## Why This Architecture

This framework is intentionally:
- **PIT-safe**: avoids forward leakage via SCD2 reconstruction
- **explainable**: kernel + convolution decomposition is easy to audit
- **extensible**: supports segment kernels, uncertainty, and executive KPI layers
- **business-ready**: naturally maps to pipeline reviews and quota conversations
