"""
data_loader.py – Load and validate wind-farm data files.

All reading logic is isolated here so tests can exercise it independently.
"""
from __future__ import annotations

import os
import io
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent          # workspace root
INTERVAL_HOURS = 1 / 6                           # 10-minute intervals → hours

TARGET_TURBINES   = ["T1", "T2", "T3"]
REFERENCE_TURBINES = ["H1", "H2", "H3", "H4", "H5", "O1", "O2"]

HIDDEN_OUTAGE_START = pd.Timestamp("2026-09-01 00:00:00")
HIDDEN_OUTAGE_END   = pd.Timestamp("2026-09-03 23:50:00")
SHOWN_TEST_START    = pd.Timestamp("2026-09-04 00:00:00")
SHOWN_TEST_END      = pd.Timestamp("2026-09-09 23:50:00")

REQUIRED_COLS   = {"turbine_name", "timestamp", "wind_speed", "test_set"}
POWER_CURVE_COLS = {"wind_speed", "power"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_excel_safe(path: str | Path | Any, **kwargs) -> pd.DataFrame:
    """Read an xlsx, xls, or csv file safely, suppressing harmless openpyxl UserWarnings."""
    filename = ""
    if isinstance(path, (str, Path)):
        filename = str(path).lower()
    elif hasattr(path, "filename"):
        filename = str(path.filename).lower()
    elif hasattr(path, "name"):
        filename = str(path.name).lower()

    if filename.endswith(".csv"):
        return pd.read_csv(path, **kwargs)

    # Attempt Excel reading, falling back to CSV if needed
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.read_excel(path, engine="openpyxl", **kwargs)
    except Exception as e:
        if isinstance(path, (str, Path)) and Path(path).exists():
            try:
                return pd.read_csv(path, **kwargs)
            except Exception:
                pass
        raise e


def _normalise_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Round noisy 10-minute timestamps to the nearest 10-minute boundary."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Round to nearest 10 minutes
    df["timestamp"] = df["timestamp"].dt.round("10min")
    return df


# ── turbine data ──────────────────────────────────────────────────────────────

def load_turbine_data(
    turbine_id: str,
    data_dir: Optional[str | Path] = None,
    file_obj=None,
) -> pd.DataFrame:
    """
    Load a turbine data file (supports both .xlsx and .csv).

    Parameters
    ----------
    turbine_id : str
        e.g. 'T1', 'H1', 'O2'
    data_dir : path, optional
        Directory to search. Defaults to DATA_DIR.
    file_obj : file-like, optional
        If provided, read from this object instead of disk.

    Returns
    -------
    DataFrame with columns: turbine_name, timestamp, wind_speed,
    wind_direction, ambient_temperature, air_pressure,
    wind_speed_stddev, test_set
    """
    if data_dir is None:
        data_dir = DATA_DIR

    if file_obj is not None:
        df = _read_excel_safe(file_obj)
    else:
        path_xlsx = Path(data_dir) / f"{turbine_id}_data.xlsx"
        path_csv  = Path(data_dir) / f"{turbine_id}_data.csv"
        if path_xlsx.exists():
            df = _read_excel_safe(path_xlsx)
        elif path_csv.exists():
            df = _read_excel_safe(path_csv)
        else:
            raise FileNotFoundError(f"Data file not found: {path_xlsx} or {path_csv}")

    # Validate columns
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {turbine_id}: {missing}")

    df = _normalise_timestamps(df)
    df["test_set"] = df["test_set"].astype(bool)

    # Keep wind_speed as float; NaN stays NaN (never coerce to 0)
    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_power_curve(
    turbine_id: str,
    data_dir: Optional[str | Path] = None,
    file_obj=None,
) -> pd.DataFrame:
    """
    Load a turbine power curve (supports both .xlsx and .csv).

    Returns a DataFrame with columns: wind_speed (m/s), power (kW).
    Values are sorted ascending by wind_speed.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    if file_obj is not None:
        df = _read_excel_safe(file_obj)
    else:
        path_xlsx = Path(data_dir) / f"{turbine_id}_power_curve.xlsx"
        path_csv  = Path(data_dir) / f"{turbine_id}_power_curve.csv"
        if path_xlsx.exists():
            df = _read_excel_safe(path_xlsx)
        elif path_csv.exists():
            df = _read_excel_safe(path_csv)
        else:
            raise FileNotFoundError(f"Power curve not found: {path_xlsx} or {path_csv}")

    missing = POWER_CURVE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing power-curve columns in {turbine_id}: {missing}")

    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
    df["power"]      = pd.to_numeric(df["power"],      errors="coerce")
    df = df.dropna(subset=["wind_speed", "power"])
    df = df.sort_values("wind_speed").reset_index(drop=True)
    return df[["wind_speed", "power"]]


# ── multi-turbine merge ───────────────────────────────────────────────────────

def load_all_references(
    data_dir: Optional[str | Path] = None,
    turbine_ids: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """Load all available reference-turbine dataframes keyed by turbine_id."""
    if data_dir is None:
        data_dir = DATA_DIR
    if turbine_ids is None:
        turbine_ids = REFERENCE_TURBINES

    result = {}
    for tid in turbine_ids:
        try:
            result[tid] = load_turbine_data(tid, data_dir=data_dir)
        except FileNotFoundError:
            pass   # silently skip missing files
    return result


def build_reference_matrix(
    ref_frames: dict[str, pd.DataFrame],
    timestamp_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align reference wind-speeds to a common timestamp index.

    Parameters
    ----------
    ref_frames : dict turbine_id → DataFrame (with 'timestamp', 'wind_speed')
    timestamp_index : DatetimeIndex
        The target timestamps to align to.

    Returns
    -------
    DataFrame indexed by timestamp, columns = turbine_ids.
    NaN where a reference is missing.
    """
    dfs = []
    for tid, df in ref_frames.items():
        s = df.set_index("timestamp")["wind_speed"].rename(tid)
        dfs.append(s)

    if not dfs:
        return pd.DataFrame(index=timestamp_index)

    ref_matrix = pd.concat(dfs, axis=1, sort=False)
    ref_matrix = ref_matrix.reindex(timestamp_index)
    return ref_matrix


def clean_reference_matrix(
    ref_matrix: pd.DataFrame,
    max_gap_intervals: int = 2,
) -> pd.DataFrame:
    """
    Handle random isolated missing values in reference wind-speed measurements.

    Per dataset documentation:
    - Random 1-2 interval gaps (10-20 min) are interpolated using time-continuity.
    - Multi-hour communication gaps (e.g. July outages) remain NaN (not fabricated).
    - Adds park-level composite signals (park_H_mean, park_O_mean) where peer
      turbines in the same wind park provide continuous signal during single-turbine outages.
    """
    cleaned = ref_matrix.copy()
    # Interpolate short random telemetry gaps (limit=2 intervals = 20 mins)
    cleaned = cleaned.interpolate(method="time", limit=max_gap_intervals)

    # Park-level aggregate signals
    h_cols = [c for c in cleaned.columns if c.startswith("H")]
    o_cols = [c for c in cleaned.columns if c.startswith("O")]

    if h_cols:
        cleaned["park_H_mean"] = cleaned[h_cols].mean(axis=1)
    if o_cols:
        cleaned["park_O_mean"] = cleaned[o_cols].mean(axis=1)

    return cleaned


def build_reference_direction_matrix(
    ref_frames: dict[str, pd.DataFrame],
    timestamp_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Extract wind direction (degrees) for reference turbines aligned to timestamp_index with short-gap interpolation."""
    dfs = []
    for tid, df in ref_frames.items():
        if "wind_direction" in df.columns:
            s = df.set_index("timestamp")["wind_direction"].rename(f"{tid}_dir")
            dfs.append(s)
    if not dfs:
        return pd.DataFrame(index=timestamp_index)
    dir_df = pd.concat(dfs, axis=1, sort=False).reindex(timestamp_index)
    # Safely interpolate short random telemetry gaps (limit=2 intervals = 20 mins)
    return dir_df.interpolate(method="time", limit=2)


# ── power curve interpolation ────────────────────────────────────────────────

def interpolate_power(wind_speeds: pd.Series, power_curve: pd.DataFrame) -> pd.Series:
    """
    Interpolate power (kW) from wind speed using the supplied power curve.

    - Clamps to curve boundaries; never silently extrapolates above max.
    - Returns NaN where wind_speed is NaN.

    Parameters
    ----------
    wind_speeds : Series of m/s values (may contain NaN)
    power_curve : DataFrame with 'wind_speed' and 'power' columns

    Returns
    -------
    Series of kW values, same index as wind_speeds.
    """
    ws_arr = wind_speeds.to_numpy(dtype=float, na_value=np.nan)
    curve_ws  = power_curve["wind_speed"].to_numpy(dtype=float)
    curve_pow = power_curve["power"].to_numpy(dtype=float)

    ws_min = curve_ws.min()
    ws_max = curve_ws.max()

    result = np.where(
        np.isnan(ws_arr),
        np.nan,
        np.interp(
            np.clip(ws_arr, ws_min, ws_max),
            curve_ws,
            curve_pow,
        ),
    )

    return pd.Series(result, index=wind_speeds.index, name="power_kw")


def compute_energy_kwh(power_kw: pd.Series, interval_hours: float = INTERVAL_HOURS) -> pd.Series:
    """Convert a power series (kW) to energy per interval (kWh)."""
    return power_kw * interval_hours


# ── outage detection ──────────────────────────────────────────────────────────

def detect_null_blocks(df: pd.DataFrame, min_block_size: int = 1) -> list[dict]:
    """
    Return a list of contiguous null-wind-speed intervals with detailed classification.

    Classifications:
    - 'Documented Period / Suspected Outage': consecutive missing >= 12 intervals
      overlapping documented test_set period (Sep 1-3).
    - 'Suspected Outage': consecutive missing >= 12 intervals (>= 2 hours).
    - 'Telemetry / Sensor Gap': consecutive missing 3 to 11 intervals (30 min - 2h).
    - 'Isolated Missing Observation': 1 to 2 intervals (10 - 20 min).
    """
    null_mask = df["wind_speed"].isna()
    groups = (null_mask != null_mask.shift()).cumsum()

    blocks = []
    for g_id, group in df[null_mask].groupby(groups[null_mask]):
        ts = group["timestamp"]
        n = len(ts)
        t_start = ts.min()
        t_end   = ts.max()
        dur_h   = n * INTERVAL_HOURS

        has_test_flag = bool(group["test_set"].any()) if "test_set" in group.columns else False
        is_documented = (t_start >= HIDDEN_OUTAGE_START and t_end <= HIDDEN_OUTAGE_END + pd.Timedelta(hours=1))

        if is_documented or (n >= 12 and has_test_flag):
            category = "Documented Period / Suspected Outage"
            confidence = "High (Documented Period)"
            rationale = (
                f"{n} contiguous missing intervals ({dur_h:.1f}h) coinciding with the "
                "documented hidden test period. Target wind speed is 100% missing."
            )
            is_candidate = True
        elif n >= 12:
            category = "Suspected Outage"
            confidence = "Medium (Suspected Outage)"
            rationale = (
                f"{n} consecutive missing intervals ({dur_h:.1f}h) exceed the 2-hour threshold. "
                "Potential localized turbine unavailability or extended communications failure."
            )
            is_candidate = True
        elif n >= 3:
            category = "Telemetry / Sensor Gap"
            confidence = "Low (Telemetry Gap)"
            rationale = (
                f"{n} consecutive missing intervals ({dur_h:.1f}h). Likely a temporary sensor drop "
                "or brief SCADA communication interruption."
            )
            is_candidate = False
        else:
            category = "Isolated Missing Observation"
            confidence = "Low (Isolated Missing)"
            rationale = f"{n} isolated missing 10-minute reading(s). Routine sensor noise."
            is_candidate = False

        blocks.append(
            {
                "start": t_start,
                "end": t_end,
                "n_intervals": n,
                "duration_hours": round(dur_h, 2),
                "duration_str": f"{int(dur_h)}h {int((dur_h % 1) * 60)}m",
                "category": category,
                "confidence": confidence,
                "rationale": rationale,
                "is_outage_candidate": is_candidate,
                "has_test_flag": has_test_flag,
            }
        )
    return sorted(blocks, key=lambda x: (not x["is_outage_candidate"], -x["n_intervals"], x["start"]))


def validate_outage_interval(
    df: pd.DataFrame,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
) -> dict:
    """
    Validate a proposed manual or automatic outage interval against the target dataset.

    Returns dict with interval statistics, validation flags, and actionable warnings.
    """
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)

    if ts_start >= ts_end:
        return {
            "valid": False,
            "error": "Outage start time must precede end time.",
            "warnings": [],
        }

    data_min = df["timestamp"].min()
    data_max = df["timestamp"].max()

    warnings_list = []
    if ts_start < data_min or ts_end > data_max:
        warnings_list.append(
            f"Interval ({ts_start.strftime('%Y-%m-%d %H:%M')} to {ts_end.strftime('%Y-%m-%d %H:%M')}) "
            f"extends outside the dataset bounds ({data_min.strftime('%Y-%m-%d')} to {data_max.strftime('%Y-%m-%d')})."
        )

    # Slice rows within interval
    in_interval = df[(df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)]
    n_expected = len(in_interval)
    n_missing  = int(in_interval["wind_speed"].isna().sum()) if n_expected > 0 else 0
    n_observed = n_expected - n_missing
    pct_missing = (n_missing / n_expected * 100.0) if n_expected > 0 else 0.0

    dur_hours = (ts_end - ts_start + pd.Timedelta(minutes=10)).total_seconds() / 3600.0

    if n_observed > 0 and n_expected > 0:
        warnings_list.append(
            f"The selected period contains {n_observed} observed target measurements ({100.0 - pct_missing:.1f}% of interval). "
            "Reconstructing this period will replace observed values with model predictions for outage estimation."
        )

    if n_expected == 0:
        warnings_list.append("No timestamps found in the dataset for the selected interval.")

    return {
        "valid": True,
        "start": str(ts_start),
        "end": str(ts_end),
        "duration_hours": round(dur_hours, 2),
        "duration_str": f"{int(dur_hours)}h {int((dur_hours % 1) * 60)}m",
        "n_expected": n_expected,
        "n_missing": n_missing,
        "n_observed": n_observed,
        "pct_missing": round(pct_missing, 1),
        "warnings": warnings_list,
    }


# ── validation helpers ────────────────────────────────────────────────────────

def get_pre_outage_mask(df: pd.DataFrame, outage_start: pd.Timestamp) -> pd.Series:
    """Boolean mask: rows strictly before outage_start with non-null wind_speed."""
    return (df["timestamp"] < outage_start) & df["wind_speed"].notna()


def get_shown_test_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: rows in the Sep 4-9 shown-test period with non-null wind_speed."""
    return (
        (df["timestamp"] >= SHOWN_TEST_START)
        & (df["timestamp"] <= SHOWN_TEST_END)
        & df["wind_speed"].notna()
    )
