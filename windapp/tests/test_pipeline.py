"""
tests/test_pipeline.py – Unit tests for data loading, metrics, pipeline integrity.

Run from project root:  python -m pytest windapp/tests/ -v
"""
import sys
import os
from pathlib import Path

# Allow importing from windapp/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from data_loader import (
    load_turbine_data,
    load_power_curve,
    interpolate_power,
    compute_energy_kwh,
    detect_null_blocks,
    get_pre_outage_mask,
    HIDDEN_OUTAGE_START,
    HIDDEN_OUTAGE_END,
    SHOWN_TEST_START,
    SHOWN_TEST_END,
    INTERVAL_HOURS,
    DATA_DIR,
)
from metrics import mae, rmse, wmape, compute_wind_metrics, rank_models
from pipeline import OutageAnalysisPipeline, chronological_split


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def t1_df():
    return load_turbine_data("T1", data_dir=DATA_DIR)

@pytest.fixture(scope="module")
def t1_curve():
    return load_power_curve("T1", data_dir=DATA_DIR)


# ── Data loading tests ────────────────────────────────────────────────────────

def test_load_t1_shape(t1_df):
    assert t1_df.shape[0] == 12960, f"Expected 12960 rows, got {t1_df.shape[0]}"
    assert t1_df.shape[1] >= 5

def test_load_t1_columns(t1_df):
    required = {"turbine_name", "timestamp", "wind_speed", "test_set"}
    assert required.issubset(set(t1_df.columns))

def test_timestamps_rounded(t1_df):
    """All timestamps must be on 10-minute boundaries."""
    ts = t1_df["timestamp"]
    bad = ts[ts.dt.second != 0]
    # After rounding, seconds should be 0
    assert len(bad) == 0, f"Found {len(bad)} timestamps with non-zero seconds"

def test_timestamps_monotone(t1_df):
    ts = t1_df["timestamp"]
    assert ts.is_monotonic_increasing, "Timestamps not sorted"

def test_hidden_outage_null(t1_df):
    """Sep 1–3 rows in T1 must all have null wind_speed."""
    mask = (t1_df["timestamp"] >= HIDDEN_OUTAGE_START) & (t1_df["timestamp"] <= HIDDEN_OUTAGE_END)
    sub = t1_df[mask]
    assert len(sub) > 0, "No rows found for hidden outage period"
    assert sub["wind_speed"].isna().all(), "Some Sep 1-3 wind speeds are non-null"

def test_shown_test_not_null(t1_df):
    """Sep 4–9 rows in T1 must all have non-null wind_speed."""
    mask = (t1_df["timestamp"] >= SHOWN_TEST_START) & (t1_df["timestamp"] <= SHOWN_TEST_END)
    sub = t1_df[mask]
    assert len(sub) == 864, f"Expected 864 shown-test rows, got {len(sub)}"
    assert sub["wind_speed"].notna().all(), "Some Sep 4-9 wind speeds are null"

def test_nan_preserved_not_zeroed(t1_df):
    """NaN values must remain as NaN, never coerced to 0."""
    null_mask = t1_df["wind_speed"].isna()
    zeros_in_null = (t1_df.loc[null_mask, "wind_speed"] == 0).sum()
    assert zeros_in_null == 0


# ── Power curve tests ─────────────────────────────────────────────────────────

def test_power_curve_columns(t1_curve):
    assert "wind_speed" in t1_curve.columns
    assert "power" in t1_curve.columns

def test_power_curve_sorted(t1_curve):
    assert t1_curve["wind_speed"].is_monotonic_increasing

def test_power_curve_length(t1_curve):
    assert len(t1_curve) >= 10, "Power curve too short"

def test_interpolate_power_basic(t1_curve):
    ws = pd.Series([0.0, 5.0, 12.0, 25.0, 30.0])  # 30 > max → should clamp
    pwr = interpolate_power(ws, t1_curve)
    assert len(pwr) == 5
    assert pwr.iloc[0] >= 0
    # 12 m/s is in the flat-top region → should be rated power
    assert pwr.iloc[2] > 0

def test_interpolate_power_nan_passthrough(t1_curve):
    ws = pd.Series([5.0, np.nan, 10.0])
    pwr = interpolate_power(ws, t1_curve)
    assert np.isnan(pwr.iloc[1])

def test_no_extrapolation_above_max(t1_curve):
    """Wind speed above curve max should be clamped, not extrapolated."""
    ws_max = t1_curve["wind_speed"].max()
    ws = pd.Series([ws_max, ws_max + 10])
    pwr = interpolate_power(ws, t1_curve)
    # Both should give same power (clamped)
    assert abs(pwr.iloc[0] - pwr.iloc[1]) < 1e-6


# ── Energy conversion tests ───────────────────────────────────────────────────

def test_energy_conversion():
    pwr = pd.Series([600.0, 1200.0, 0.0])
    energy = compute_energy_kwh(pwr)
    # 600 kW × (1/6) h = 100 kWh
    assert abs(energy.iloc[0] - 100.0) < 1e-9
    assert abs(energy.iloc[1] - 200.0) < 1e-9
    assert abs(energy.iloc[2] - 0.0)   < 1e-9

def test_energy_interval_hours():
    assert abs(INTERVAL_HOURS - 1/6) < 1e-12


# ── Metric tests ──────────────────────────────────────────────────────────────

def test_mae_basic():
    a = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 2.0])
    assert abs(mae(a, p) - 2/3) < 1e-9

def test_rmse_basic():
    a = np.array([0.0, 0.0])
    p = np.array([1.0, 1.0])
    assert abs(rmse(a, p) - 1.0) < 1e-9

def test_wmape_definition():
    # WMAPE = sum(|pred-actual|) / sum(|actual|) * 100
    a = np.array([2.0, 4.0])
    p = np.array([3.0, 3.0])
    # |3-2| + |3-4| = 2; |2| + |4| = 6; → 2/6 * 100 = 33.33%
    assert abs(wmape(a, p) - 100/3) < 1e-6

def test_wmape_undefined_near_zero():
    a = np.array([0.0, 0.0])
    p = np.array([1.0, 2.0])
    assert np.isnan(wmape(a, p))

def test_metrics_nan_handling():
    a = np.array([1.0, np.nan, 3.0])
    p = np.array([1.0, 2.0,   3.0])
    m = compute_wind_metrics(a, p)
    assert m["n"] == 2
    assert abs(m["mae"] - 0.0) < 1e-9

def test_metrics_all_nan():
    a = np.array([np.nan, np.nan])
    p = np.array([1.0, 2.0])
    m = compute_wind_metrics(a, p)
    assert m["n"] == 0
    assert np.isnan(m["mae"])


# ── Model ranking tests ───────────────────────────────────────────────────────

def test_rank_models_basic():
    metrics = {
        "ModelA": {"available": True, "n": 100, "mae": 1.0, "rmse": 1.5, "wmape": 10.0, "n_refs": 3},
        "ModelB": {"available": True, "n": 100, "mae": 0.8, "rmse": 1.2, "wmape": 8.0,  "n_refs": 3},
        "ModelC": {"available": False, "n": 0,  "mae": np.nan, "rmse": np.nan, "wmape": np.nan, "n_refs": 0},
    }
    ranked = rank_models(metrics)
    assert ranked[0] == "ModelB"   # best on all three
    assert "ModelC" not in ranked  # excluded (unavailable)

def test_rank_models_empty():
    ranked = rank_models({})
    assert ranked == []

def test_rank_models_all_unavailable():
    metrics = {
        "M": {"available": False, "n": 0, "mae": np.nan, "rmse": np.nan, "wmape": np.nan, "n_refs": 0}
    }
    assert rank_models(metrics) == []


# ── Chronological validation ──────────────────────────────────────────────────

def test_chronological_split_order():
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=100, freq="10min"),
                       "wind_speed": np.random.rand(100)})
    train, val = chronological_split(df, val_frac=0.2)
    assert len(train) == 80
    assert len(val)   == 20
    assert train["timestamp"].max() < val["timestamp"].min()

def test_chronological_split_no_shuffle():
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=10, freq="10min"),
                       "a": range(10)})
    train, val = chronological_split(df, val_frac=0.3)
    assert list(train["a"]) == list(range(7))
    assert list(val["a"])   == list(range(7, 10))


# ── Leakage prevention tests ─────────────────────────────────────────────────

def test_pre_outage_mask(t1_df):
    mask = get_pre_outage_mask(t1_df, HIDDEN_OUTAGE_START)
    sub  = t1_df[mask]
    assert (sub["timestamp"] < HIDDEN_OUTAGE_START).all(), "Pre-outage mask includes outage rows"
    assert sub["wind_speed"].notna().all(), "Pre-outage mask includes null wind speeds"

def test_no_outage_labels_in_pre_outage(t1_df):
    """Verify no Sep 1+ rows appear in pre-outage data."""
    mask = get_pre_outage_mask(t1_df, HIDDEN_OUTAGE_START)
    sub  = t1_df[mask]
    assert (sub["timestamp"] < HIDDEN_OUTAGE_START).all()

def test_detect_null_blocks_finds_outage(t1_df):
    blocks = detect_null_blocks(t1_df)
    large  = [b for b in blocks if b["n_intervals"] >= 400]
    assert len(large) >= 1, "Expected at least one large null block (hidden outage)"
    block = large[0]
    assert block["start"] == HIDDEN_OUTAGE_START or str(block["start"]) >= "2026-09-01"


# ── Full pipeline smoke test (runs with real data) ────────────────────────────

def test_pipeline_runs():
    """Full pipeline run – validates no crash and basic output integrity."""
    pipe = OutageAnalysisPipeline(target_id="T1", data_dir=DATA_DIR)
    pipe.run()

    assert pipe.selected_model_name is not None, "No model selected"
    assert pipe.val_metrics.get("n", 0) > 0, "No validation samples"
    assert pipe.corr_df is not None and len(pipe.corr_df) > 0
    assert pipe.outage_df is not None, "No outage predictions generated"
    assert not np.isnan(pipe.total_energy_kwh), "Total energy is NaN"
    assert pipe.total_energy_kwh > 0, "Total energy should be positive"

def test_pipeline_energy_consistency():
    """Interval sum must equal total_energy_kwh."""
    pipe = OutageAnalysisPipeline(target_id="T1", data_dir=DATA_DIR)
    pipe.run()
    if pipe.outage_df is not None:
        interval_sum = pipe.outage_df["energy_kwh"].sum()
        assert abs(interval_sum - pipe.total_energy_kwh) < 0.1, \
            f"Interval sum {interval_sum} ≠ total {pipe.total_energy_kwh}"

def test_shown_test_metrics_computed():
    pipe = OutageAnalysisPipeline(target_id="T1", data_dir=DATA_DIR)
    pipe.run()
    assert pipe.shown_test_metrics is not None
    assert pipe.shown_test_metrics.get("n", 0) > 0, "No shown-test samples evaluated"

def test_no_future_leakage():
    """Val timestamps must all be < outage start."""
    pipe = OutageAnalysisPipeline(target_id="T1", data_dir=DATA_DIR)
    pipe.run()
    assert pipe.val_end < HIDDEN_OUTAGE_START, "Validation overlaps with outage period"

def test_correlation_pre_outage_only(t1_df):
    """Correlation must be computed only on pre-outage observations."""
    from pipeline import compute_correlations, build_reference_matrix
    from data_loader import load_all_references
    ref_frames = load_all_references(data_dir=DATA_DIR)
    target_series = t1_df.set_index("timestamp")["wind_speed"]
    ref_matrix = build_reference_matrix(ref_frames, pd.DatetimeIndex(t1_df["timestamp"]))
    corr_df = compute_correlations(target_series, ref_matrix, HIDDEN_OUTAGE_START)
    # All correlations should be non-empty for the big reference turbines
    assert len(corr_df) > 0
    assert "pearson" in corr_df.columns


def test_detect_null_blocks_categories(t1_df):
    """Ensure detect_null_blocks properly categorizes outages vs telemetry gaps."""
    blocks = detect_null_blocks(t1_df)
    categories = [b["category"] for b in blocks]
    assert any("Documented Period" in c or "Suspected Outage" in c for c in categories)
    # 432 block should be high confidence outage candidate
    outage_candidates = [b for b in blocks if b.get("is_outage_candidate")]
    assert len(outage_candidates) >= 1
    main_outage = outage_candidates[0]
    assert main_outage["n_intervals"] == 432
    assert "High" in main_outage["confidence"]


def test_validate_outage_interval_valid(t1_df):
    """Validate valid hidden outage interval."""
    from data_loader import validate_outage_interval
    res = validate_outage_interval(t1_df, "2026-09-01 00:00", "2026-09-03 23:50")
    assert res["valid"] is True
    assert res["n_expected"] == 432
    assert res["n_missing"] == 432
    assert res["pct_missing"] == 100.0


def test_validate_outage_interval_invalid(t1_df):
    """Validate invalid date range where start is after end."""
    from data_loader import validate_outage_interval
    res = validate_outage_interval(t1_df, "2026-09-05 00:00", "2026-09-02 00:00")
    assert res["valid"] is False
    assert "error" in res
    assert "precede" in res["error"].lower() or "start" in res["error"].lower()


def test_validate_outage_interval_observed_warning(t1_df):
    """Validate warning when interval contains observed data."""
    from data_loader import validate_outage_interval
    res = validate_outage_interval(t1_df, "2026-09-04 00:00", "2026-09-06 00:00")
    assert res["valid"] is True
    assert res["n_observed"] > 0
    assert len(res["warnings"]) > 0


def test_diagnose_wind_errors():
    """Verify diagnose_wind_errors structure and computation."""
    from metrics import diagnose_wind_errors
    np.random.seed(42)
    actual = np.array([3.0, 5.0, 7.5, 9.0, 11.0, 14.0, 16.0, 8.0, 4.0, 6.0])
    predicted = actual + np.array([-0.2, 0.5, -0.3, 0.8, -0.5, 0.1, -0.2, 0.4, -0.1, 0.3])
    ts = pd.date_range("2026-09-04 00:00", periods=10, freq="10min")
    res = diagnose_wind_errors(actual, predicted, timestamps=ts)

    assert "mean_bias" in res
    assert "speed_bins" in res
    assert len(res["speed_bins"]) > 0
    assert "dynamics" in res
    assert "top_errors" in res
    assert len(res["top_errors"]) <= 10


def test_bias_metric():
    """Verify bias metric."""
    from metrics import bias
    actual = np.array([5.0, 6.0, 7.0])
    pred = np.array([6.0, 7.0, 8.0])
    assert abs(bias(actual, pred) - 1.0) < 1e-6


def test_flask_api_routes():
    """Verify new Flask API endpoints return valid JSON."""
    from app import app
    client = app.test_client()

    # /api/outage/detect
    r_det = client.get("/api/outage/detect?target_id=T1")
    assert r_det.status_code == 200
    det_json = r_det.get_json()
    assert det_json["status"] == "ok"
    assert "candidates" in det_json

    # /api/outage/validate
    r_val = client.post("/api/outage/validate", json={
        "target_id": "T1",
        "start": "2026-09-01 00:00",
        "end": "2026-09-03 23:50"
    })
    assert r_val.status_code == 200
    val_json = r_val.get_json()
    assert val_json["status"] == "ok"
    assert val_json["result"]["valid"] is True

    # /api/outage/confirm
    r_conf = client.post("/api/outage/confirm", json={
        "target_id": "T1",
        "start": "2026-09-01 00:00",
        "end": "2026-09-03 23:50",
        "notes": "Test confirmation"
    })
    assert r_conf.status_code == 200
    conf_json = r_conf.get_json()
    assert conf_json["status"] == "ok"
    assert conf_json["outage_confirmed"] is True


def test_csv_file_loading(tmp_path):
    """Verify CSV files can be loaded by load_turbine_data and load_power_curve."""
    # Create temporary CSV file matching turbine schema
    csv_file = tmp_path / "T1_data.csv"
    df_sample = pd.DataFrame({
        "turbine_name": ["T1", "T1"],
        "timestamp": ["2026-07-01 00:00:00", "2026-07-01 00:10:00"],
        "wind_speed": [5.2, 5.8],
        "test_set": [False, False],
    })
    df_sample.to_csv(csv_file, index=False)
    loaded = load_turbine_data("T1", file_obj=str(csv_file))
    assert len(loaded) == 2
    assert "wind_speed" in loaded.columns

