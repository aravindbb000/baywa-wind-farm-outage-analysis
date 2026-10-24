"""
models.py – Wind-speed prediction models for outage estimation.

Implements four model families:
  1. ARIMA/SARIMA  (statsmodels)
  2. XGBoost
  3. LightGBM
  4. Random Forest (scikit-learn)

Each model follows a common interface:
  fit(X_train, y_train) → self
  predict(X_pred) → np.ndarray

Chronological validation is enforced by the caller (pipeline.py).
No target labels from the outage or after are ever used in training.
"""
from __future__ import annotations

import warnings
import traceback
from typing import Optional, Any

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# ── model status ─────────────────────────────────────────────────────────────

class ModelResult:
    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.trained   = False
        self.error_msg: Optional[str] = None
        self.model: Any = None
        self.feature_names: list[str] = []
        self.reference_turbines: list[str] = []
        self.n_train: int = 0

    def mark_unavailable(self, msg: str):
        self.available = False
        self.error_msg = msg

    def mark_trained(self, model, feature_names: list[str], n_train: int):
        self.available = True
        self.trained   = True
        self.model = model
        self.feature_names = feature_names
        self.n_train = n_train

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ── feature engineering ───────────────────────────────────────────────────────

def build_tree_features(
    ref_df: pd.DataFrame,
    ref_cols: list[str],
    lags: list[int] = [1, 2, 3],
    dir_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build feature matrix for XGBoost/LightGBM/RF from reference wind speeds and environmental cues.

    Features:
    - Current reference wind speeds (ref_cols)
    - Lagged reference wind speeds (lags × ref_cols)
    - Hour of day, minute, day of week
    - Cyclical diurnal harmonics (sin_hour, cos_hour)
    - Directional orthogonal components (sin_dir, cos_dir) when available

    Parameters
    ----------
    ref_df : DataFrame indexed by timestamp, columns = reference turbine IDs
    ref_cols : which reference columns to use
    lags : past-only lags in number of 10-minute intervals
    dir_df : Optional DataFrame with wind direction columns ({tid}_dir)

    Returns
    -------
    DataFrame of features, same index as ref_df.
    """
    feats = pd.DataFrame(index=ref_df.index)

    for col in ref_cols:
        if col in ref_df.columns:
            feats[col] = ref_df[col]
            for lag in lags:
                feats[f"{col}_lag{lag}"] = ref_df[col].shift(lag)

    if hasattr(ref_df.index, 'hour'):
        feats["hour"]       = ref_df.index.hour
        feats["minute"]     = ref_df.index.minute
        feats["day_of_week"] = ref_df.index.day_of_week if hasattr(ref_df.index, 'day_of_week') else ref_df.index.dayofweek

        # Continuous cyclical diurnal cycle (preserves smooth transition across 23:50 -> 00:00)
        frac_hour = ref_df.index.hour + ref_df.index.minute / 60.0
        feats["sin_hour"] = np.sin(2 * np.pi * frac_hour / 24.0)
        feats["cos_hour"] = np.cos(2 * np.pi * frac_hour / 24.0)

    # Directional circular decomposition if direction data is provided
    if dir_df is not None and not dir_df.empty:
        for col in ref_cols:
            dir_col = f"{col}_dir"
            if dir_col in dir_df.columns:
                rad = np.radians(dir_df[dir_col])
                feats[f"{col}_sin_dir"] = np.sin(rad)
                feats[f"{col}_cos_dir"] = np.cos(rad)

    return feats


# ── ARIMA/SARIMA ──────────────────────────────────────────────────────────────

class ARIMAModel(ModelResult):
    """
    Univariate ARIMA(2,1,1) on the target wind-speed series.
    Uses only pre-outage target measurements.
    Forecasts forward (open-loop); no reference turbines used.
    """
    ORDER = (2, 1, 1)

    def __init__(self):
        super().__init__("ARIMA/SARIMA")
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            self._SARIMAX = SARIMAX
            self.available = True
        except ImportError as e:
            self.mark_unavailable(f"statsmodels not available: {e}")

    def fit(self, y_train: pd.Series | np.ndarray) -> "ARIMAModel":
        if not self.available:
            return self
        try:
            y_arr = np.asarray(y_train, dtype=float)
            y_arr = y_arr[~np.isnan(y_arr)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = self._SARIMAX(
                    y_arr,
                    order=self.ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self._result = mod.fit(disp=False, maxiter=50)
            self.n_train = len(y_arr)
            self.trained = True
            self.feature_names = ["target_wind_speed (lagged)"]
            self.reference_turbines = []
        except Exception as e:
            self.mark_unavailable(f"ARIMA fit failed: {e}")
        return self

    def predict_n(self, n_steps: int) -> np.ndarray:
        """Forecast n_steps beyond training period."""
        if not self.trained:
            return np.full(n_steps, np.nan)
        try:
            forecast = self._result.forecast(steps=n_steps)
            return np.maximum(np.asarray(forecast, dtype=float), 0.0)
        except Exception:
            return np.full(n_steps, np.nan)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ARIMA uses predict_n for sequential forecast")


# ── Tree-based models ─────────────────────────────────────────────────────────

class XGBoostModel(ModelResult):
    """XGBoost regressor using lagged reference wind speeds as features."""

    def __init__(self):
        super().__init__("XGBoost")
        try:
            import xgboost as xgb
            self._xgb = xgb
            self.available = True
        except ImportError as e:
            self.mark_unavailable(f"xgboost not available: {e}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: list[str], ref_turbines: list[str]) -> "XGBoostModel":
        if not self.available:
            return self
        try:
            model = self._xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                verbosity=0,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            self.model = model
            self.n_train = len(y_train)
            self.trained = True
            self.feature_names = feature_names
            self.reference_turbines = ref_turbines
        except Exception as e:
            self.mark_unavailable(f"XGBoost fit failed: {e}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.full(len(X), np.nan)
        return np.maximum(self.model.predict(X), 0.0)


class LightGBMModel(ModelResult):
    """LightGBM regressor using lagged reference wind speeds as features."""

    def __init__(self):
        super().__init__("LightGBM")
        try:
            import lightgbm as lgb
            self._lgb = lgb
            self.available = True
        except ImportError as e:
            self.mark_unavailable(f"lightgbm not available: {e}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: list[str], ref_turbines: list[str]) -> "LightGBMModel":
        if not self.available:
            return self
        try:
            model = self._lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                verbosity=-1,
                n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            self.model = model
            self.n_train = len(y_train)
            self.trained = True
            self.feature_names = feature_names
            self.reference_turbines = ref_turbines
        except Exception as e:
            self.mark_unavailable(f"LightGBM fit failed: {e}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.full(len(X), np.nan)
        return np.maximum(self.model.predict(X), 0.0)


class RandomForestModel(ModelResult):
    """Random Forest regressor using lagged reference wind speeds as features."""

    def __init__(self):
        super().__init__("Random Forest")
        try:
            from sklearn.ensemble import RandomForestRegressor
            self._RF = RandomForestRegressor
            self.available = True
        except ImportError as e:
            self.mark_unavailable(f"scikit-learn not available: {e}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: list[str], ref_turbines: list[str]) -> "RandomForestModel":
        if not self.available:
            return self
        try:
            model = self._RF(
                n_estimators=100,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            self.model = model
            self.n_train = len(y_train)
            self.trained = True
            self.feature_names = feature_names
            self.reference_turbines = ref_turbines
        except Exception as e:
            self.mark_unavailable(f"RandomForest fit failed: {e}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.full(len(X), np.nan)
        return np.maximum(self.model.predict(X), 0.0)

