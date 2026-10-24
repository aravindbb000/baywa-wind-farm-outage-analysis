"""
metrics.py – Metric calculations for wind-speed and energy evaluation.

All metrics are computed from actual data arrays; nothing is hard-coded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ── wind-speed metrics ────────────────────────────────────────────────────────

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error (m/s)."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - predicted[mask])))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error (m/s)."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


def wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error (%).

    WMAPE = sum(|predicted - actual|) / sum(|actual|) × 100

    Returns np.nan when sum(|actual|) is near zero (undefined).
    """
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return np.nan
    denom = np.sum(np.abs(actual[mask]))
    if denom < 1e-9:
        return np.nan
    return float(np.sum(np.abs(predicted[mask] - actual[mask])) / denom * 100)


def valid_sample_count(actual: np.ndarray, predicted: np.ndarray) -> int:
    """Number of paired non-NaN samples."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    return int(mask.sum())


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean prediction bias: mean(predicted - actual). Positive = overprediction, Negative = underprediction."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(predicted[mask] - actual[mask]))


def compute_wind_metrics(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict:
    """Return a dict with MAE, RMSE, WMAPE, Bias, and valid sample count."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return {
        "mae":   mae(a, p),
        "rmse":  rmse(a, p),
        "wmape": wmape(a, p),
        "bias":  bias(a, p),
        "n":     valid_sample_count(a, p),
    }


# ── detailed deviation diagnostics ──────────────────────────────────────────

def diagnose_wind_errors(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    timestamps: pd.Series | None = None,
) -> dict:
    """
    Produce structured error diagnostics for wind-speed predictions.
    Identifies residual patterns, speed-bin performance, dynamics, and extreme errors.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p))

    if mask.sum() == 0:
        return {}

    a_valid = a[mask]
    p_valid = p[mask]
    residuals = p_valid - a_valid
    abs_errors = np.abs(residuals)

    # Error distribution stats
    mean_bias = float(np.mean(residuals))
    median_res = float(np.median(residuals))
    std_res = float(np.std(residuals))
    p10, p25, p75, p90 = [float(x) for x in np.percentile(residuals, [10, 25, 75, 90])]

    # Histogram of residuals (15 bins)
    hist_counts, bin_edges = np.histogram(residuals, bins=15)
    hist_data = [
        {
            "bin_start": round(float(bin_edges[i]), 2),
            "bin_end": round(float(bin_edges[i+1]), 2),
            "bin_center": round(float((bin_edges[i] + bin_edges[i+1]) / 2.0), 2),
            "count": int(hist_counts[i]),
        }
        for i in range(len(hist_counts))
    ]

    # Error by wind-speed bins
    bins_def = [
        ("<4 m/s (Low / Cut-in)", 0.0, 4.0),
        ("4-8 m/s (Moderate)", 4.0, 8.0),
        ("8-12 m/s (Rated Ramp)", 8.0, 12.0),
        (">12 m/s (Rated / High)", 12.0, 99.0),
    ]

    speed_bins = []
    for label, low, high in bins_def:
        b_mask = (a_valid >= low) & (a_valid < high) if high < 90 else (a_valid >= low)
        n_b = int(b_mask.sum())
        if n_b > 0:
            a_b = a_valid[b_mask]
            p_b = p_valid[b_mask]
            res_b = p_b - a_b
            denom = float(np.sum(np.abs(a_b)))
            wmape_b = float(np.sum(np.abs(res_b)) / denom * 100.0) if denom > 1e-6 else np.nan
            speed_bins.append({
                "range": label,
                "n": n_b,
                "pct_of_total": round(n_b / len(a_valid) * 100.0, 1),
                "mae": round(float(np.mean(np.abs(res_b))), 3),
                "rmse": round(float(np.sqrt(np.mean(res_b ** 2))), 3),
                "bias": round(float(np.mean(res_b)), 3),
                "wmape": round(wmape_b, 2) if not np.isnan(wmape_b) else None,
            })

    # Error during dynamic vs stable periods (using 1-step wind delta)
    dynamics = {}
    if len(a_valid) > 1:
        delta_actual = np.abs(np.diff(a_valid, prepend=a_valid[0]))
        stable_mask = delta_actual < 0.5
        dynamic_mask = delta_actual >= 0.5

        if stable_mask.sum() > 0:
            dynamics["stable"] = {
                "n": int(stable_mask.sum()),
                "mae": round(float(np.mean(abs_errors[stable_mask])), 3),
                "rmse": round(float(np.sqrt(np.mean(residuals[stable_mask] ** 2))), 3),
            }
        if dynamic_mask.sum() > 0:
            dynamics["dynamic"] = {
                "n": int(dynamic_mask.sum()),
                "mae": round(float(np.mean(abs_errors[dynamic_mask])), 3),
                "rmse": round(float(np.sqrt(np.mean(residuals[dynamic_mask] ** 2))), 3),
            }

    # Top 10 largest error timestamps
    top_errors = []
    if timestamps is not None and len(timestamps) == len(a):
        ts_valid = pd.Series(timestamps).iloc[np.where(mask)[0]].reset_index(drop=True)
        top_indices = np.argsort(abs_errors)[::-1][:10]
        for idx in top_indices:
            top_errors.append({
                "timestamp": str(ts_valid.iloc[idx]),
                "actual_ms": round(float(a_valid[idx]), 2),
                "predicted_ms": round(float(p_valid[idx]), 2),
                "abs_error_ms": round(float(abs_errors[idx]), 2),
                "residual_ms": round(float(residuals[idx]), 2),
                "error_type": "Overprediction" if residuals[idx] > 0 else "Underprediction",
            })

    return {
        "n_samples": int(len(a_valid)),
        "mean_bias": round(mean_bias, 3),
        "median_residual": round(median_res, 3),
        "std_residual": round(std_res, 3),
        "percentiles": {
            "p10": round(p10, 3),
            "p25": round(p25, 3),
            "p75": round(p75, 3),
            "p90": round(p90, 3),
        },
        "histogram": hist_data,
        "speed_bins": speed_bins,
        "dynamics": dynamics,
        "top_errors": top_errors,
        "tendency": "Slight Underprediction" if mean_bias < -0.05 else ("Slight Overprediction" if mean_bias > 0.05 else "Well-Centered (Unbiased)"),
    }


# ── power metrics ─────────────────────────────────────────────────────────────

def power_mae(actual_kw: np.ndarray, predicted_kw: np.ndarray) -> float:
    mask = ~(np.isnan(actual_kw) | np.isnan(predicted_kw))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual_kw[mask] - predicted_kw[mask])))


def power_rmse(actual_kw: np.ndarray, predicted_kw: np.ndarray) -> float:
    mask = ~(np.isnan(actual_kw) | np.isnan(predicted_kw))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual_kw[mask] - predicted_kw[mask]) ** 2)))


def power_wmape(actual_kw: np.ndarray, predicted_kw: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error (%) for turbine power.
    WMAPE = sum(|predicted_kw - actual_kw|) / sum(|actual_kw|) * 100
    """
    mask = ~(np.isnan(actual_kw) | np.isnan(predicted_kw))
    if mask.sum() == 0:
        return np.nan
    denom = np.sum(np.abs(actual_kw[mask]))
    if denom < 1e-9:
        return np.nan
    return float(np.sum(np.abs(predicted_kw[mask] - actual_kw[mask])) / denom * 100)


def compute_power_metrics(
    actual_kw: np.ndarray | pd.Series,
    predicted_kw: np.ndarray | pd.Series,
) -> dict:
    """Return dict with power MAE (kW), RMSE (kW), WMAPE (%), and sample count."""
    a = np.asarray(actual_kw, dtype=float)
    p = np.asarray(predicted_kw, dtype=float)
    return {
        "mae":   power_mae(a, p),
        "rmse":  power_rmse(a, p),
        "wmape": power_wmape(a, p),
        "bias":  bias(a, p),
        "n":     valid_sample_count(a, p),
    }


def energy_deviation_pct(actual_kwh: float, predicted_kwh: float) -> float:
    """Percentage deviation of predicted from actual total energy."""
    if abs(actual_kwh) < 1e-6:
        return np.nan
    return (predicted_kwh - actual_kwh) / abs(actual_kwh) * 100


# ── model ranking ─────────────────────────────────────────────────────────────

def rank_models(model_metrics: dict[str, dict]) -> list[str]:
    """
    Deterministic model selection by mean rank across MAE, RMSE, WMAPE.

    Selection rule (documented):
    1. Exclude models with fewer than 1 valid sample or any NaN metric.
    2. Rank each eligible model on MAE, RMSE, WMAPE (lower = better rank 1).
    3. Compute mean rank across the three metrics.
    4. Tie-breakers (in order):
       a. More metric wins (most times ranked #1).
       b. Lower MAE.
       c. Fixed model order: LightGBM → Random Forest → XGBoost → ARIMA/SARIMA → Ridge Regression → SBA → Historical Mean.
       d. Fewer reference turbines.
    5. Return ordered list of model names (best first).

    Parameters
    ----------
    model_metrics : {model_name: {"mae": float, "rmse": float, "wmape": float,
                                   "n": int, "n_refs": int, "available": bool}}
    """
    FIXED_ORDER = [
        "LightGBM",
        "Random Forest",
        "XGBoost",
        "ARIMA/SARIMA",
    ]

    eligible = {
        name: m for name, m in model_metrics.items()
        if m.get("available", False)
        and m.get("n", 0) >= 1
        and not any(np.isnan(m.get(k, np.nan)) for k in ["mae", "rmse", "wmape"])
    }

    if not eligible:
        return []

    names = list(eligible.keys())

    # Build rank arrays
    def rank_by(key, reverse=False):
        vals = [(eligible[n][key], n) for n in names]
        vals.sort(key=lambda x: x[0], reverse=reverse)
        return {n: i + 1 for i, (_, n) in enumerate(vals)}

    mae_ranks   = rank_by("mae")
    rmse_ranks  = rank_by("rmse")
    wmape_ranks = rank_by("wmape")

    mean_ranks = {
        n: (mae_ranks[n] + rmse_ranks[n] + wmape_ranks[n]) / 3
        for n in names
    }

    # Count metric wins (times ranked #1)
    wins = {n: sum(1 for r in [mae_ranks[n], rmse_ranks[n], wmape_ranks[n]] if r == 1)
            for n in names}

    def sort_key(n):
        fixed_pos = FIXED_ORDER.index(n) if n in FIXED_ORDER else 999
        return (
            mean_ranks[n],
            -wins[n],
            eligible[n].get("mae", 999),
            fixed_pos,
            eligible[n].get("n_refs", 999),
        )

    ranked = sorted(names, key=sort_key)
    return ranked
