"""
pipeline.py – End-to-end analysis pipeline.

Orchestrates data loading, reference selection, model training,
chronological validation, shown-test evaluation, outage prediction,
and energy calculation.

Key leakage-prevention guarantees:
- Pre-outage training: only target rows with timestamp < outage_start
- Correlation: only paired pre-outage rows
- Validation holdout: chronological split within pre-outage data
- No target labels from outage or test period used anywhere in fitting/tuning
"""
from __future__ import annotations

import warnings
from datetime import timedelta
from typing import Optional, Any

import numpy as np
import pandas as pd
from scipy import stats

from data_loader import (
    load_turbine_data,
    load_power_curve,
    load_all_references,
    build_reference_matrix,
    clean_reference_matrix,
    build_reference_direction_matrix,
    detect_null_blocks,
    interpolate_power,
    compute_energy_kwh,
    get_pre_outage_mask,
    get_shown_test_mask,
    INTERVAL_HOURS,
    DATA_DIR,
    HIDDEN_OUTAGE_START,
    HIDDEN_OUTAGE_END,
    SHOWN_TEST_START,
    SHOWN_TEST_END,
)
from models import (
    ARIMAModel,
    XGBoostModel,
    LightGBMModel,
    RandomForestModel,
    build_tree_features,
    RANDOM_SEED,
)
from metrics import (
    compute_wind_metrics,
    diagnose_wind_errors,
    power_mae,
    power_rmse,
    energy_deviation_pct,
    rank_models,
)

LAGS = [1, 2, 3]   # 10-, 20-, 30-minute lags for tree models
VAL_FRAC = 0.20    # chronological validation fraction of pre-outage data


# ── correlation ───────────────────────────────────────────────────────────────

def compute_correlations(
    target_series: pd.Series,
    ref_matrix: pd.DataFrame,
    outage_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations strictly using pre-outage paired observations.

    Parameters
    ----------
    target_series : Series indexed by timestamp
    ref_matrix : DataFrame indexed by timestamp, columns = ref turbine IDs
    outage_start : timestamp marking start of outage (exclusive boundary)

    Returns
    -------
    DataFrame with columns: turbine, pearson, spearman, n_paired
    sorted descending by pearson.
    """
    pre_mask = target_series.index < outage_start
    y = target_series[pre_mask].dropna()

    rows = []
    for col in ref_matrix.columns:
        ref_s = ref_matrix.loc[pre_mask, col]
        paired = pd.concat([y, ref_s], axis=1, sort=False).dropna()
        n = len(paired)
        if n < 3:
            rows.append({"turbine": col, "pearson": np.nan, "spearman": np.nan, "n_paired": n})
            continue
        a, b = paired.iloc[:, 0].to_numpy(), paired.iloc[:, 1].to_numpy()
        try:
            r_p, _ = stats.pearsonr(a, b)
            r_s, _ = stats.spearmanr(a, b)
        except Exception:
            r_p, r_s = np.nan, np.nan
        rows.append({"turbine": col, "pearson": r_p, "spearman": r_s, "n_paired": n})

    df = pd.DataFrame(rows).sort_values("pearson", ascending=False).reset_index(drop=True)
    return df


# ── reference selection ───────────────────────────────────────────────────────

def select_reference_turbines(
    corr_df: pd.DataFrame,
    min_pearson: float = 0.3,
    max_refs: int = 7,
) -> list[str]:
    """
    Select reference turbines with Pearson correlation >= min_pearson,
    ordered by correlation descending. Returns at most max_refs.
    """
    eligible = corr_df[corr_df["pearson"].notna() & (corr_df["pearson"] >= min_pearson)]
    return eligible["turbine"].head(max_refs).tolist()


# ── chronological validation ──────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (train, val) keeping chronological order."""
    n = len(df)
    split_idx = int(n * (1 - val_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


# ── full pipeline ─────────────────────────────────────────────────────────────

class OutageAnalysisPipeline:
    """
    Full analysis pipeline for one target turbine and one outage interval.

    Attributes set after run():
    - corr_df: correlation table
    - ref_turbines: selected reference turbine IDs
    - model_results: dict {model_name: ModelResult}
    - model_metrics: dict {model_name: metric_dict}
    - ranked_models: ordered list of model names (best first)
    - selected_model_name: str
    - val_metrics: dict for selected model (validation)
    - shown_test_metrics: dict for selected model (shown test)
    - outage_df: DataFrame with timestamp, predicted_wind_speed, predicted_power_kw, energy_kwh
    - shown_test_df: DataFrame with timestamp, actual, predicted wind_speed
    - total_energy_kwh: float
    - outage_start, outage_end: Timestamps
    - n_outage_intervals: int
    - val_start, val_end: Timestamps for validation window
    - n_val: int
    - power_curve: DataFrame
    - status_messages: list of warning/info strings
    """

    def __init__(
        self,
        target_id: str = "T1",
        outage_start: Optional[pd.Timestamp] = None,
        outage_end: Optional[pd.Timestamp] = None,
        data_dir=None,
        assume_zero_production: bool = True,
    ):
        self.target_id = target_id
        self.outage_start = outage_start or HIDDEN_OUTAGE_START
        self.outage_end   = outage_end   or HIDDEN_OUTAGE_END
        self.data_dir     = data_dir or DATA_DIR
        self.assume_zero_production = assume_zero_production
        self.status_messages: list[str] = []
        self._ran = False

    # ── internal helpers ───────────────────────────────────────────────────────

    def _warn(self, msg: str):
        self.status_messages.append(msg)

    def _build_tree_XY(
        self,
        target_series: pd.Series,
        ref_matrix: pd.DataFrame,
        ref_cols: list[str],
        mask: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build (X, y, feature_names) for tree models from a boolean mask."""
        feats = build_tree_features(ref_matrix, ref_cols, lags=LAGS, dir_df=getattr(self, "_dir_matrix", None))
        df_combined = pd.concat([target_series.rename("target"), feats], axis=1, sort=False)
        df_combined = df_combined[mask]
        df_combined = df_combined.dropna()
        X = df_combined[feats.columns].to_numpy()
        y = df_combined["target"].to_numpy()
        return X, y, list(feats.columns)

    def _predict_tree(
        self,
        model_result,
        ref_matrix: pd.DataFrame,
        ref_cols: list[str],
        pred_index: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Generate predictions for pred_index using tree model, computing lags with full continuous context."""
        if not model_result.trained:
            return np.full(len(pred_index), np.nan)
        feats_full = build_tree_features(ref_matrix, ref_cols, lags=LAGS, dir_df=getattr(self, "_dir_matrix", None))
        feats_pred = feats_full.loc[pred_index]
        # Impute any residual isolated gap so no outage interval is lost
        if feats_pred.isna().any().any():
            feats_pred = feats_pred.ffill().bfill()
        X = feats_pred.to_numpy()
        preds = model_result.predict(X)
        return preds

    # ── main run ───────────────────────────────────────────────────────────────

    def run(self) -> "OutageAnalysisPipeline":
        """Execute the full pipeline. Returns self."""
        self._ran = True
        self.status_messages = []

        # 1. Load target data and power curve
        target_df = load_turbine_data(self.target_id, data_dir=self.data_dir)
        self.power_curve = load_power_curve(self.target_id, data_dir=self.data_dir)

        # 2. Load reference data
        ref_frames = load_all_references(data_dir=self.data_dir)
        if not ref_frames:
            raise RuntimeError("No reference turbine data found.")

        # 3. Build aligned reference matrix over full date range
        target_ts_index = pd.DatetimeIndex(target_df["timestamp"])
        target_series   = target_df.set_index("timestamp")["wind_speed"]
        raw_ref_matrix  = build_reference_matrix(ref_frames, target_ts_index)
        # Handle random missing values in references and compute park-level composite signals
        ref_matrix      = clean_reference_matrix(raw_ref_matrix, max_gap_intervals=2)
        self._dir_matrix = build_reference_direction_matrix(ref_frames, target_ts_index)

        # 4. Compute correlations (strictly pre-outage, no leakage)
        self.corr_df = compute_correlations(
            target_series, ref_matrix, self.outage_start
        )
        self.ref_turbines = select_reference_turbines(self.corr_df)
        if not self.ref_turbines:
            self._warn("No reference turbines met the minimum correlation threshold.")
            self.ref_turbines = self.corr_df["turbine"].head(3).tolist()

        # 5. Pre-outage training mask
        pre_outage_mask = (target_series.index < self.outage_start) & target_series.notna()

        # 6. Chronological validation split within pre-outage data
        pre_outage_data = target_df[target_df["timestamp"] < self.outage_start].copy()
        pre_outage_data = pre_outage_data[pre_outage_data["wind_speed"].notna()]
        train_df, val_df = chronological_split(pre_outage_data, val_frac=VAL_FRAC)

        self.val_start = val_df["timestamp"].min() if len(val_df) else None
        self.val_end   = val_df["timestamp"].max() if len(val_df) else None
        self.n_val     = len(val_df)

        val_mask   = pd.Series(False, index=target_ts_index)
        train_mask = pd.Series(False, index=target_ts_index)
        for ts in val_df["timestamp"]:
            if ts in val_mask.index:
                val_mask[ts] = True
        for ts in train_df["timestamp"]:
            if ts in train_mask.index:
                train_mask[ts] = True

        # 7. Train all models on training portion, validate on val portion
        self.model_results: dict[str, Any] = {}
        self.model_metrics: dict[str, dict] = {}

        ref_cols = self.ref_turbines

        # --- ARIMA/SARIMA ---
        arima = ARIMAModel()
        arima_val_preds = np.full(self.n_val, np.nan)
        if arima.available and len(train_df) > 20:
            y_train_arima = train_df.set_index("timestamp")["wind_speed"]
            arima.fit(y_train_arima)
            if arima.trained:
                arima_val_preds = arima.predict_n(self.n_val)
        self.model_results["ARIMA/SARIMA"] = arima
        y_val = val_df["wind_speed"].to_numpy()
        m = compute_wind_metrics(y_val, arima_val_preds)
        m["available"] = arima.available and arima.trained
        m["n_refs"] = 0
        m["error_msg"] = arima.error_msg
        if not m["available"]:
            m["mae"] = np.nan; m["rmse"] = np.nan; m["wmape"] = np.nan
        self.model_metrics["ARIMA/SARIMA"] = m

        # --- XGBoost ---
        xgb_model = XGBoostModel()
        if xgb_model.available and len(train_df) > 20 and ref_cols:
            X_tr, y_tr, feat_names = self._build_tree_XY(
                target_series, ref_matrix, ref_cols,
                train_mask & target_series.notna()
            )
            if len(y_tr) > 0:
                xgb_model.fit(X_tr, y_tr, feat_names, ref_cols)
        self.model_results["XGBoost"] = xgb_model
        if xgb_model.trained:
            val_index = pd.DatetimeIndex(val_df["timestamp"])
            xgb_val_preds = self._predict_tree(xgb_model, ref_matrix, ref_cols, val_index)
            m2 = compute_wind_metrics(y_val, xgb_val_preds)
        else:
            m2 = {"mae": np.nan, "rmse": np.nan, "wmape": np.nan, "n": 0}
        m2["available"] = xgb_model.trained
        m2["n_refs"] = len(ref_cols)
        m2["error_msg"] = xgb_model.error_msg
        self.model_metrics["XGBoost"] = m2

        # --- LightGBM ---
        lgb_model = LightGBMModel()
        if lgb_model.available and len(train_df) > 20 and ref_cols:
            X_tr, y_tr, feat_names = self._build_tree_XY(
                target_series, ref_matrix, ref_cols,
                train_mask & target_series.notna()
            )
            if len(y_tr) > 0:
                lgb_model.fit(X_tr, y_tr, feat_names, ref_cols)
        self.model_results["LightGBM"] = lgb_model
        if lgb_model.trained:
            val_index = pd.DatetimeIndex(val_df["timestamp"])
            lgb_val_preds = self._predict_tree(lgb_model, ref_matrix, ref_cols, val_index)
            m3 = compute_wind_metrics(y_val, lgb_val_preds)
        else:
            m3 = {"mae": np.nan, "rmse": np.nan, "wmape": np.nan, "n": 0}
        m3["available"] = lgb_model.trained
        m3["n_refs"] = len(ref_cols)
        m3["error_msg"] = lgb_model.error_msg
        self.model_metrics["LightGBM"] = m3

        # --- Random Forest ---
        rf_model = RandomForestModel()
        if rf_model.available and len(train_df) > 20 and ref_cols:
            X_tr, y_tr, feat_names = self._build_tree_XY(
                target_series, ref_matrix, ref_cols,
                train_mask & target_series.notna()
            )
            if len(y_tr) > 0:
                rf_model.fit(X_tr, y_tr, feat_names, ref_cols)
        self.model_results["Random Forest"] = rf_model
        if rf_model.trained:
            val_index = pd.DatetimeIndex(val_df["timestamp"])
            rf_val_preds = self._predict_tree(rf_model, ref_matrix, ref_cols, val_index)
            m4 = compute_wind_metrics(y_val, rf_val_preds)
        else:
            m4 = {"mae": np.nan, "rmse": np.nan, "wmape": np.nan, "n": 0}
        m4["available"] = rf_model.trained
        m4["n_refs"] = len(ref_cols)
        m4["error_msg"] = rf_model.error_msg
        self.model_metrics["Random Forest"] = m4

        # --- Baseline Comparison (Historical Mean on Validation Window) ---
        y_train_mean = float(np.mean(train_df["wind_speed"]))
        baseline_preds = np.full(self.n_val, y_train_mean)
        self.baseline_metrics = compute_wind_metrics(y_val, baseline_preds)

        # 8. Rank models and select best
        self.ranked_models = rank_models(self.model_metrics)
        if not self.ranked_models:
            self._warn("No eligible model could be trained. Check data and dependencies.")
            self.selected_model_name = None
            self.val_metrics = {}
        else:
            self.selected_model_name = self.ranked_models[0]
            self.val_metrics = self.model_metrics[self.selected_model_name]

        # 9. Shown-test evaluation (Sep 4-9, observed target readings)
        shown_test_df = target_df[
            (target_df["timestamp"] >= SHOWN_TEST_START)
            & (target_df["timestamp"] <= SHOWN_TEST_END)
            & target_df["wind_speed"].notna()
        ].copy()

        self.shown_test_df = None
        self.shown_test_metrics = {}
        self.diagnostics = {}

        if len(shown_test_df) > 0 and self.selected_model_name:
            st_index = pd.DatetimeIndex(shown_test_df["timestamp"])
            model = self.model_results[self.selected_model_name]

            if self.selected_model_name == "ARIMA/SARIMA" and model.trained:
                # For ARIMA, forecast from end of training
                all_pre = target_df[
                    (target_df["timestamp"] < SHOWN_TEST_START) & target_df["wind_speed"].notna()
                ].set_index("timestamp")["wind_speed"]
                arima2 = ARIMAModel()
                arima2.fit(all_pre)
                if arima2.trained:
                    st_preds = arima2.predict_n(len(shown_test_df))
                else:
                    st_preds = np.full(len(shown_test_df), np.nan)
            elif model.trained:
                st_preds = self._predict_tree(model, ref_matrix, ref_cols, st_index)
            else:
                st_preds = np.full(len(shown_test_df), np.nan)

            shown_test_df = shown_test_df.copy()
            shown_test_df["predicted_wind_speed"] = st_preds
            shown_test_df["actual_wind_speed"]    = shown_test_df["wind_speed"]

            # Compute power comparisons
            actual_power = interpolate_power(shown_test_df["actual_wind_speed"],   self.power_curve)
            pred_power   = interpolate_power(shown_test_df["predicted_wind_speed"], self.power_curve)
            shown_test_df["actual_power_kw"]    = actual_power.values
            shown_test_df["predicted_power_kw"] = pred_power.values

            y_st_actual = shown_test_df["actual_wind_speed"].to_numpy()
            y_st_pred   = shown_test_df["predicted_wind_speed"].to_numpy()
            self.shown_test_metrics = compute_wind_metrics(y_st_actual, y_st_pred)

            self.shown_test_metrics["power_mae"]  = power_mae(actual_power.to_numpy(), pred_power.to_numpy())
            self.shown_test_metrics["power_rmse"] = power_rmse(actual_power.to_numpy(), pred_power.to_numpy())

            actual_total_kwh = float(compute_energy_kwh(actual_power).sum())
            pred_total_kwh   = float(compute_energy_kwh(pred_power).sum())
            self.shown_test_metrics["energy_deviation_pct"] = energy_deviation_pct(
                actual_total_kwh, pred_total_kwh
            )
            self.shown_test_df = shown_test_df

            # Diagnostic breakdown
            self.diagnostics = diagnose_wind_errors(
                y_st_actual,
                y_st_pred,
                timestamps=shown_test_df["timestamp"],
            )

        # 10. Outage-period predictions (hidden Sep 1-3 or user-selected)
        outage_df = target_df[
            (target_df["timestamp"] >= self.outage_start)
            & (target_df["timestamp"] <= self.outage_end)
        ].copy()

        self.n_outage_intervals = len(outage_df)
        self.outage_df = None
        self.total_energy_kwh = np.nan
        self.detected_blocks = detect_null_blocks(target_df)

        if len(outage_df) > 0 and self.selected_model_name:
            out_index = pd.DatetimeIndex(outage_df["timestamp"])
            model = self.model_results[self.selected_model_name]

            if self.selected_model_name == "ARIMA/SARIMA" and model.trained:
                # Re-fit ARIMA on all pre-outage data for actual outage forecast
                all_pre2 = target_df[
                    (target_df["timestamp"] < self.outage_start) & target_df["wind_speed"].notna()
                ].set_index("timestamp")["wind_speed"]
                arima3 = ARIMAModel()
                arima3.fit(all_pre2)
                if arima3.trained:
                    out_preds = arima3.predict_n(len(outage_df))
                else:
                    out_preds = np.full(len(outage_df), np.nan)
            elif model.trained:
                out_preds = self._predict_tree(model, ref_matrix, ref_cols, out_index)
            else:
                out_preds = np.full(len(outage_df), np.nan)

            outage_df = outage_df.copy()
            outage_df["predicted_wind_speed"] = out_preds
            pred_power_out = interpolate_power(
                pd.Series(out_preds, name="wind_speed"), self.power_curve
            )
            outage_df["predicted_power_kw"] = pred_power_out.values
            outage_df["energy_kwh"] = compute_energy_kwh(pred_power_out).values
            outage_df["cumulative_energy_kwh"] = outage_df["energy_kwh"].cumsum()

            self.outage_df = outage_df
            valid_energy = outage_df["energy_kwh"].dropna()
            self.total_energy_kwh = float(valid_energy.sum()) if len(valid_energy) > 0 else np.nan

        return self

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def selected_model(self):
        return self.model_results.get(self.selected_model_name)

    @property
    def outage_duration_str(self) -> str:
        delta = self.outage_end - self.outage_start + pd.Timedelta(minutes=10)
        hours = int(delta.total_seconds() // 3600)
        mins  = int((delta.total_seconds() % 3600) // 60)
        return f"{hours}h {mins}m"

    def to_summary_dict(self) -> dict:
        """Compact summary for the Overview KPI card."""
        stm = self.shown_test_metrics or {}
        base = self.baseline_metrics or {}

        return {
            "target_id": self.target_id,
            "selected_model": self.selected_model_name or "N/A",
            "ref_turbines": self.ref_turbines,
            "val_mae":   self.val_metrics.get("mae",   np.nan),
            "val_rmse":  self.val_metrics.get("rmse",  np.nan),
            "val_wmape": self.val_metrics.get("wmape", np.nan),
            "val_bias":  self.val_metrics.get("bias",  np.nan),
            "n_val":     self.val_metrics.get("n",     0),
            "baseline_val_mae": base.get("mae", np.nan),
            "baseline_val_rmse": base.get("rmse", np.nan),
            "baseline_val_wmape": base.get("wmape", np.nan),
            # Shown test metrics
            "st_mae": stm.get("mae", np.nan),
            "st_rmse": stm.get("rmse", np.nan),
            "st_wmape": stm.get("wmape", np.nan),
            "st_power_mae": stm.get("power_mae", np.nan),
            "st_power_rmse": stm.get("power_rmse", np.nan),
            "st_energy_deviation_pct": stm.get("energy_deviation_pct", np.nan),
            "total_energy_kwh": self.total_energy_kwh,
            "outage_start": str(self.outage_start),
            "outage_end":   str(self.outage_end),
            "outage_duration_str": self.outage_duration_str,
            "n_outage_intervals": self.n_outage_intervals,
            "assume_zero_production": self.assume_zero_production,
            "status_messages": self.status_messages,
        }
