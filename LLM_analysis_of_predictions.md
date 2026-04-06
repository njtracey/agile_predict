# Agile Predict — LLM Analysis Guide

This document provides everything an LLM needs to analyse the agile_predict model's performance. It covers the database schema, how to run the analysis script, how to interpret the output, and template prompts for common analysis tasks.

## Database Schema

The SQLite database (`db.sqlite3`) contains 6 tables. All timestamps are UTC.

### PriceHistory (`prices_pricehistory`)

Actual half-hourly electricity prices from the Octopus Agile tariff (region F).

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| date_time | DATETIME (unique) | Settlement period start (UTC) |
| day_ahead | FLOAT | Wholesale day-ahead price (£/MWh) |
| agile | FLOAT | Consumer Agile tariff price (p/kWh) |

This is the ground truth. Every prediction is compared against `agile` from this table.

### Forecasts (`prices_forecasts`)

One row per model training run. Typically one run per day around 16:15 UK time.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| name | VARCHAR(64) unique | Timestamp string, e.g. "2026-04-06 16:15" |
| created_at | DATETIME | When the forecast was created |
| mean | FLOAT | Cross-validation RMSE (£/MWh day-ahead) |
| stdev | FLOAT | Cross-validation RMSE standard deviation |

The `mean` field is the ensemble cross-validation RMSE in £/MWh (day-ahead space, not Agile p/kWh). Lower is better. Multiply by ~0.21 to approximate the Agile-space error.

### ForecastData (`prices_forecastdata`)

Raw input features for each forecast run — the weather/demand/generation data the model used.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| forecast_id | FK → Forecasts | Which forecast run |
| date_time | DATETIME | Target settlement period (UTC) |
| day_ahead | FLOAT (nullable) | Predicted day-ahead price (£/MWh) |
| bm_wind | FLOAT | Metered wind generation (MW) |
| solar | FLOAT | Embedded solar generation (MW) |
| emb_wind | FLOAT | Embedded wind generation (MW) |
| temp_2m | FLOAT | Temperature at 2m (°C) |
| wind_10m | FLOAT | Wind speed at 10m (m/s) |
| rad | FLOAT | Direct solar radiation (W/m²) |
| demand | FLOAT | National demand (MW) |

### AgileData (`prices_agiledata`)

Our model's predictions converted to Agile tariff prices, per region.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| forecast_id | FK → Forecasts | Which forecast run |
| region | CHAR(1) | DNO region code (A-P, X=national) |
| date_time | DATETIME | Target settlement period (UTC) |
| agile_pred | FLOAT | Predicted Agile price (p/kWh) |
| agile_low | FLOAT | Lower confidence band (p/kWh) |
| agile_high | FLOAT | Upper confidence band (p/kWh) |

Region F (North Eastern England) is the local region and is kept permanently. Other regions are trimmed after 14 days.

### OfficialAgileData (`prices_officialagiledata`)

Predictions from the official agilepredict.com website, fetched at each update run for comparison.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| forecast_id | FK → Forecasts | Linked to same forecast run as our predictions |
| date_time | DATETIME | Target settlement period (UTC) |
| agile_pred | FLOAT | Official predicted Agile price (p/kWh) |
| agile_low | FLOAT | Official lower confidence band (p/kWh) |
| agile_high | FLOAT | Official upper confidence band (p/kWh) |

Unique constraint: `(forecast_id, date_time)`. This table may be empty if the official API was unavailable or if Phase 0 was recently deployed.

### History (`prices_history`)

Historical weather and generation data used for model training.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| date_time | DATETIME (unique) | Settlement period start (UTC) |
| total_wind | FLOAT | Total wind generation (MW) |
| bm_wind | FLOAT | Metered wind generation (MW) |
| solar | FLOAT | Embedded solar generation (MW) |
| temp_2m | FLOAT | Temperature at 2m (°C) |
| wind_10m | FLOAT | Wind speed at 10m (m/s) |
| rad | FLOAT | Direct solar radiation (W/m²) |
| demand | FLOAT | National demand (MW) |

### Relationships

```
Forecasts (1) ──→ (many) ForecastData      (input features per slot)
Forecasts (1) ──→ (many) AgileData          (our predictions per slot per region)
Forecasts (1) ──→ (many) OfficialAgileData  (official predictions per slot)
```

All three child tables CASCADE on Forecasts deletion.

PriceHistory and History are standalone — no foreign keys. PriceHistory is the ground truth for accuracy measurement.


## Running the Analysis Script

```bash
cd agile_predict
venv/bin/python bin/analyze_model_performance.py [--days 30] [--format json]
```

Arguments:
- `--days N` — Analysis window in days (default: 30). Use 7 for a quick check, 90 for a longer trend.
- `--format json|text` — Output format. JSON (default) is best for LLM parsing. Text is human-readable.

The script connects directly to `db.sqlite3` using pure sqlite3 + pandas. No Django dependency, no venv activation beyond the Python binary.

## Interpreting the JSON Output

### `metadata`

```json
{
  "script_version": "1.0.0",
  "run_timestamp": "2026-04-06T16:30:00+00:00",
  "database_path": "/path/to/db.sqlite3",
  "analysis_window_days": 30
}
```

Check `script_version` to know which metrics are available. The `analysis_window_days` tells you how far back the analysis looks.

### `database_summary`

Record counts and date ranges for every table. Key things to check:
- `price_history.count` — should be growing by ~48 rows/day (one per half-hour)
- `forecasts.count` — one per day (after cleanup dedup)
- `forecasts.with_training_scores` — how many forecasts have cross-validation RMSE recorded
- `agile_data.region_f_count` — our predictions for region F
- `official_agile_data.count` — official predictions (0 if Phase 0 is new)

### `our_model_accuracy`

Our model's predictions compared against actual Agile prices from PriceHistory.

```json
{
  "overall": {"mae": 5.67, "rmse": 8.03, "bias": 2.19, "coverage_pct": 42.9, "n": 4415},
  "by_horizon": {
    "0-12h":  {"mae": 0.05, "rmse": 0.08, "bias": 0.01, "coverage_pct": 95.0, "n": 500},
    "12-24h": {"mae": 0.30, "rmse": 0.45, "bias": 0.05, "coverage_pct": 90.0, "n": 600},
    "24-48h": {"mae": 3.10, "rmse": 4.50, "bias": 1.20, "coverage_pct": 75.0, "n": 800},
    "2-4d":   {"mae": 4.40, "rmse": 6.20, "bias": 1.80, "coverage_pct": 65.0, "n": 700},
    "4-7d":   {"mae": 7.30, "rmse": 9.80, "bias": 2.50, "coverage_pct": 55.0, "n": 900},
    "7-14d":  {"mae": 7.10, "rmse": 9.50, "bias": 2.30, "coverage_pct": 50.0, "n": 915}
  }
}
```

Field definitions:
- `mae` — Mean Absolute Error (p/kWh). The average magnitude of prediction errors.
- `rmse` — Root Mean Squared Error (p/kWh). Penalises large errors more than MAE.
- `bias` — Mean signed error (p/kWh). Positive = model over-predicts on average.
- `coverage_pct` — Percentage of actuals falling within the `[agile_low, agile_high]` confidence band. Target: ~80% for a well-calibrated 10th-90th percentile interval.
- `n` — Number of matched prediction-actual pairs used.

If a horizon has `"status": "insufficient_data"`, there aren't enough matched pairs (minimum 5) to compute reliable metrics.

### `official_model_accuracy`

Same structure as `our_model_accuracy`, but computed from OfficialAgileData. If the table is empty, you'll see:

```json
{"status": "no_data", "note": "OfficialAgileData not yet populated"}
```

### `head_to_head`

Direct comparison where both models predicted the same target slot from the same forecast run.

```json
{
  "total_matched_pairs": 3200,
  "by_horizon": {
    "24-48h": {"our_mae": 3.10, "official_mae": 5.50, "delta_mae": -2.40, "winner": "ours", "n": 400}
  }
}
```

- `delta_mae` — Negative means our model is better (lower MAE). Positive means official is better.
- `winner` — "ours", "official", or "tie".

### `prediction_convergence`

Shows how prediction accuracy improves as the target time approaches. Each horizon bucket shows the MAE for predictions made at that lead time.

- A steep drop from 7-14d to 0-12h indicates the model improves significantly as more information becomes available.
- Flat convergence suggests the model isn't effectively using near-term information.

### `training_state`

```json
{
  "rmse_trend": [
    {"forecast_name": "2026-04-06 16:15", "rmse": 3.45, "stdev": 0.82},
    {"forecast_name": "2026-04-05 16:15", "rmse": 3.52, "stdev": 0.79}
  ],
  "training_data_volume": 137952,
  "forecast_data_features": ["day_ahead", "bm_wind", "solar", "emb_wind", "temp_2m", "wind_10m", "rad", "demand"]
}
```

- `rmse_trend` — Cross-validation RMSE over recent runs (£/MWh day-ahead space). Should be stable or decreasing.
- `training_data_volume` — Total ForecastData rows. Grows by ~600/day.
- `forecast_data_features` — Columns stored in ForecastData (a subset of what the model actually uses — runtime features like lags and volatility are computed on the fly).

### `actual_price_statistics`

Daily summaries of actual Agile prices. Useful for context — if prices were unusually volatile, expect higher MAE.

```json
{
  "recent_days": [
    {"date": "2026-04-05", "mean": 22.3, "min": 1.2, "max": 41.7, "stdev": 8.9, "slots": 48}
  ]
}
```

## Performance Benchmarks

Use these benchmarks to assess model quality at each horizon. All values are MAE in p/kWh.

| Horizon | Excellent | Good | Acceptable | Poor |
|---------|-----------|------|------------|------|
| 0-12h | < 0.1 | < 0.5 | < 1.0 | > 1.0 |
| 12-24h | < 0.5 | < 1.0 | < 2.0 | > 2.0 |
| 24-48h | < 2.0 | < 4.0 | < 6.0 | > 6.0 |
| 2-4d | < 3.0 | < 5.0 | < 7.0 | > 7.0 |
| 4-7d | < 5.0 | < 7.0 | < 10.0 | > 10.0 |
| 7-14d | < 5.0 | < 7.0 | < 10.0 | > 10.0 |

Context for these benchmarks:
- 0-12h slots are mostly filled with actual published Agile prices, so MAE should be near zero.
- 12-24h slots often include day-ahead auction results (known wholesale prices), so MAE should be very low.
- 24-48h is the first true prediction horizon — this is where model quality matters most.
- Beyond 4 days, weather forecast uncertainty dominates and MAE naturally increases.

Coverage rate benchmarks (for the confidence bands):
- Target: 78-82% (for 10th-90th percentile bands)
- Good: 70-85%
- Poor: < 60% (bands too narrow) or > 95% (bands too wide, not useful)

## Agile Pricing Formula

The Octopus Agile tariff price is derived from the wholesale day-ahead price:

```
agile_price (p/kWh) = day_ahead_price (£/MWh) × factor + peak_adder
```

For Region F (North Eastern England):
- `factor = 0.21`
- `peak_adder = 12` (applied during 16:00-19:00 UK time only, 0 otherwise)

Example:
- Day-ahead = £50/MWh, off-peak: `50 × 0.21 + 0 = 10.5 p/kWh`
- Day-ahead = £50/MWh, peak: `50 × 0.21 + 12 = 22.5 p/kWh`

Each DNO region has its own factor and peak adder. Region F is the local deployment region. The model trains on day-ahead prices (£/MWh) and converts to Agile (p/kWh) at the output stage using these factors.

## Template Prompts

Copy and paste these into an LLM session to trigger specific analyses.

### Quick Health Check

```
Run the analysis script and give me a summary of how the model is performing:

cd agile_predict && venv/bin/python bin/analyze_model_performance.py --days 7

Tell me:
1. Is the model healthy? (check RMSE trend is stable)
2. What's the MAE at each horizon?
3. Are the confidence bands well-calibrated? (coverage should be 78-82%)
4. Any anomalies in recent actual prices?
```

### Head-to-Head vs Official

```
Run the analysis script and compare our model against the official agilepredict.com:

cd agile_predict && venv/bin/python bin/analyze_model_performance.py --days 30

Focus on the head_to_head section. For each horizon bucket, tell me:
- Which model wins and by how much (delta_mae)?
- Where are we strongest and weakest?
- Overall, are we beating the official model?
```

### Deep Dive on a Specific Horizon

```
Run the analysis script with a 30-day window:

cd agile_predict && venv/bin/python bin/analyze_model_performance.py --days 30

I'm interested in the 24-48h horizon specifically. Tell me:
- What's our MAE and RMSE?
- What's the bias? (are we systematically over or under-predicting?)
- How does our coverage compare to the target 80%?
- How does this compare to the official model at the same horizon?
```

### Trend Analysis

```
Run the analysis script twice with different windows to see if we're improving:

cd agile_predict && venv/bin/python bin/analyze_model_performance.py --days 7
cd agile_predict && venv/bin/python bin/analyze_model_performance.py --days 30

Compare the 7-day vs 30-day MAE at each horizon. If the 7-day MAE is lower, we're improving. If higher, we may be regressing. Also check the training_state.rmse_trend for the cross-validation RMSE over recent runs.
```

### Full Report

```
Run the analysis script and produce a comprehensive report:

cd agile_predict && venv/bin/python bin/analyze_model_performance.py --days 30 --format text

Summarise the key findings:
1. Database health (record counts, any gaps)
2. Our model accuracy by horizon
3. Comparison with official model (if data available)
4. Confidence band calibration quality
5. Training state (RMSE trend, data volume)
6. Recent price volatility context
7. Recommendations for improvement
```

## Enhancement Roadmap

All 8 phases of model enhancements are implemented and deployed:

| Phase | Description | Status | Impact |
|-------|-------------|--------|--------|
| 0 | Official predictions storage | Deployed | Enables ongoing comparison with agilepredict.com |
| 1 | Lag prices, volatility, residual demand features | Deployed | Adds autoregressive and regime-awareness features |
| 2 | Gas/carbon prices, interconnectors, auction data | Deployed | Adds fundamental price drivers and known auction results |
| 3 | XGBoost + LightGBM ensemble | Deployed | Combines complementary model strengths |
| 4 | Horizon-specific models (near/medium/far) | Deployed | Specialised models for each prediction horizon |
| 5 | Adaptive training window | Deployed | Shorter window during volatile periods |
| 6 | Conformal prediction calibration | Deployed | Properly calibrated confidence bands (target 80% coverage) |
| 7 | Analysis tooling and this documentation | Deployed | LLM-friendly performance analysis |

### What Each Phase Contributes

- Phases 1-2 added 12 new features (lag prices, volatility, residual demand, gas, carbon, system prices, CCGT generation, CCGT share, interconnector flows). The model now has 23 features total.
- Phase 3 replaced the single XGBoost model with a weighted XGBoost + LightGBM ensemble. Weights are determined by inverse cross-validation RMSE.
- Phase 4 trains 3 separate ensemble models for near (0-24h), medium (24-48h), and far (48h+) horizons. Each specialises in the patterns most relevant to its time range.
- Phase 5 automatically shortens the training window to 60 days when 30-day price volatility exceeds 1.5× the 180-day baseline.
- Phase 6 uses conformal prediction to widen the confidence bands based on held-out calibration residuals, targeting 78-82% coverage.
- Phase 7 provides this documentation and the `analyze_model_performance.py` script.

### Data Sources

| Source | Data | Update Frequency |
|--------|------|-----------------|
| Octopus Energy API | Agile tariff prices (actual) | Every 30 min |
| Nord Pool / N2EX | Day-ahead wholesale prices | Daily ~16:00 |
| Elexon BMRS MID | Day-ahead auction results | Daily ~12:00 |
| Elexon BMRS | System prices, CCGT generation, interconnector flows | Half-hourly |
| NESO | Wind/solar/demand forecasts and outturn | Half-hourly |
| Open-Meteo | Temperature, wind speed, solar radiation | Hourly |
| OilPriceAPI | UK gas price, EU carbon price | Daily (cached) |
| agilepredict.com | Official model predictions | Per forecast run |
