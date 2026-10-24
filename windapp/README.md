# BayWa r.e. Wind Farm Outage Analysis — Local App

A portable, self-contained Flask web application for the BayWa r.e. wind-farm
outage hackathon.  All calculations run locally using the supplied data files.

---

## Quick Start (Windows)

```powershell
# 1. Install dependencies
python -m pip install -r windapp\requirements.txt

# 2. Start the app
python windapp\app.py
```

Then open **http://localhost:5000** in your browser.

---

## App Structure

Two main pages:

### 1. Wind Farm Outage Analysis (`/`)

Section tabs:
| Tab | Contents |
|---|---|
| **Overview** | KPI cards: selected model, validation MAE/RMSE/WMAPE, estimated energy, ref turbines |
| **Data Upload** | Load workspace data or upload xlsx files; validation messages |
| **Outage Detection** | Missing-interval detection, outage selector, observed + predicted time series, CSV download |
| **Explainability** | Pearson/Spearman correlation bars, model rationale, validation window summary |
| **Energy Loss** | Power curve → kWh conversion, power and cumulative-energy charts, key-insight cards |
| **Metrics** | Selected model, shown-test KPI cards, actual-vs-predicted chart, CSV download |
| **Report** | Report preview, download HTML, download interval CSV |

**Advanced button** (top-right) opens a drawer with:
- Full model leaderboard (all 4 families, per-model metrics, rank, tie-break explanation)
- Detailed correlation table with paired sample counts
- Chronological validation window details and coverage
- Technical model configuration and data cutoffs
- Methodological limitations

### 2. Insurance Report (`/insurance`)

Standalone report page using the same pipeline results:
- Outage period KPI cards
- Wind-speed and cumulative-energy charts
- Selected model and validation evidence
- Shown-test performance (clearly separate from validation)
- Correlation evidence
- Assumptions and limitations
- Download HTML and CSV

---

## Modelling

### Models compared

| Model | Implementation | Type |
|---|---|---|
| ARIMA/SARIMA | statsmodels SARIMAX(2,1,2)(1,0,1,144) | Univariate |
| XGBoost | xgboost XGBRegressor | Multivariate |
| LightGBM | lightgbm LGBMRegressor | Multivariate |
| Random Forest | sklearn RandomForestRegressor | Multivariate |

Tree models use:
- Current reference wind speeds (H1–H5, O1–O2 filtered by pre-outage Pearson ≥ 0.3)
- 10-, 20-, 30-minute lagged past-only reference wind speeds
- Hour of day, minute, day of week

### Selection rule (deterministic)

1. Rank each eligible model equally on wind-speed MAE, RMSE, and WMAPE.
2. Compute mean rank across the three metrics.
3. Tie-breakers: (a) more metric wins, (b) lower MAE, (c) fixed order
   ARIMA/SARIMA → Random Forest → XGBoost → LightGBM, (d) fewer references.
4. Unavailable or failed models are excluded and disclosed.

### Leakage prevention

- Training: only pre-outage target data (strictly before Sep 1, 00:00)
- Correlation: pre-outage paired observations only
- Validation: chronological hold-out (last 20% of pre-outage data)
- No target labels from Sep 1 onward are used in training, tuning, feature selection, or correlation

### Periods

| Period | Dates | Status |
|---|---|---|
| Training | 2026-07-01 – ~2026-08-28 | Pre-outage, non-null |
| Validation | ~2026-08-28 – 2026-08-31 | Chronological hold-out |
| Hidden outage | 2026-09-01 – 2026-09-03 | Target wind speed missing; predicted |
| Shown test | 2026-09-04 – 2026-09-09 | Observed actuals; independent evaluation |

---

## Metric Definitions

| Metric | Formula |
|---|---|
| MAE | mean(|predicted - actual|) |
| RMSE | sqrt(mean((predicted - actual)²)) |
| WMAPE | Σ\|pred−actual\| / Σ\|actual\| × 100 % |
| Energy deviation | (predicted_total − actual_total) / \|actual_total\| × 100 % |

**WMAPE** is undefined (reported as N/A) when Σ|actual| is near zero.

---

## Energy Calculation

1. Predict wind speed for each outage interval.
2. Interpolate power (kW) from the turbine-specific power curve (linear interpolation, clamped at boundaries).
3. Energy per interval = power (kW) × 1/6 h (10-minute interval).
4. Total = sum over all outage intervals.

> ⚠️ **Disclaimer**: Expected normal energy is potential loss only under the zero-production assumption.
> It is **not** confirmed lost energy or a verified insurance claim amount.

---

## Data Format

All xlsx files share this schema:

| Column | Type | Notes |
|---|---|---|
| turbine_name | str | e.g. T1, H1 |
| timestamp | datetime | 10-min intervals, UTC-naive |
| wind_speed | float | m/s; NaN where missing |
| wind_direction | float | degrees |
| ambient_temperature | float | °C |
| air_pressure | float | hPa |
| wind_speed_stddev | float | m/s |
| test_set | bool | True = Sep 1–9 |

Power curve schema: `turbine_name`, `wind_speed` (m/s), `power` (kW).

---

## Running Tests

```powershell
cd "c:\Users\ARAVIND R\Downloads\data_hackathon_baywa_re (1)"
python -m pytest windapp\tests\ -v --timeout=300
```

**Fast unit tests** (no pipeline, ~6 s):
```powershell
python -m pytest windapp\tests\ -v -k "not pipeline and not shown_test and not correlation_pre and not future"
```

---

## Docker (optional)

```bash
docker build -t baywa-outage windapp/
docker run -p 5000:5000 -v "$(pwd):/data" baywa-outage
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `python -m pip install -r windapp\requirements.txt` |
| Port 5000 in use | Edit `app.py` → change `port=5000` |
| ARIMA very slow | SARIMA with seasonal order (1,0,1,144) trains on ~8 k rows; allow 60–120 s |
| App shows "not_run" | Click **Run Analysis** on the Overview tab first |
| Upload fails | Filename must start with turbine ID (e.g. `T1_data.xlsx`) |

---

## Limitations

- Prediction uncertainty is not calibrated; no confidence intervals.
- Reference turbines are from different parks; local effects differ.
- Wind direction, temperature, pressure not used as model features.
- Power conversion uses the supplied curve; actual output may differ.
- ARIMA/SARIMA is univariate and ignores reference turbines.
- Energy totals assume zero production during the outage.
