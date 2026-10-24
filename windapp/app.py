"""
app.py – BayWa r.e. Wind Farm Outage Analysis — Flask application.

Two main pages:
  /          → Wind Farm Outage Analysis (tabs: Overview, Data Upload,
                Outage Detection, Explainability, Energy Loss, Metrics, Report)
  /insurance → Insurance Report

Advanced panel is served via /api/advanced and embedded in the main page.
All data APIs return JSON consumed by the frontend JavaScript.
"""
from __future__ import annotations

import io
import os
import sys
import json
import csv
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from flask import (
    Flask,
    jsonify,
    request,
    render_template,
    send_file,
    abort,
    Response,
)

# ── path setup ────────────────────────────────────────────────────────────────
APP_DIR  = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

from data_loader import (
    load_turbine_data,
    load_power_curve,
    load_all_references,
    build_reference_matrix,
    detect_null_blocks,
    validate_outage_interval,
    interpolate_power,
    compute_energy_kwh,
    DATA_DIR,
    TARGET_TURBINES,
    REFERENCE_TURBINES,
    HIDDEN_OUTAGE_START,
    HIDDEN_OUTAGE_END,
    SHOWN_TEST_START,
    SHOWN_TEST_END,
    INTERVAL_HOURS,
)
from pipeline import OutageAnalysisPipeline
from metrics import compute_wind_metrics, power_mae, power_rmse, energy_deviation_pct

# ── app setup ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB upload limit

# ── session state (in-memory, single user) ───────────────────────────────────
# In production this would be a proper cache/session; for hackathon local use
# a module-level dict is fine.
_state: dict = {
    "data_dir":          str(ROOT_DIR),
    "target_id":         "T1",
    "outage_start":      str(HIDDEN_OUTAGE_START),
    "outage_end":        str(HIDDEN_OUTAGE_END),
    "outage_notes":      "Documented hidden test period (Sep 1-3, 2026)",
    "outage_confirmed":  True,
    "pipeline":          None,
    "pipeline_stale":    False,
    "uploaded_files":    {},   # turbine_id → path
    "assume_zero":       True,
}


def _safe_float(v) -> float | None:
    """Return Python float or None for JSON serialisation."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f   # NaN → None
    except Exception:
        return None


def _fmt(v, decimals=3) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    return f"{v:.{decimals}f}"


def _series_to_json(ts: pd.Series, val: pd.Series) -> list[dict]:
    """Zip two series into [{x: iso_ts, y: value}, …]."""
    out = []
    for t, v in zip(ts, val):
        y = None if (v is None or (isinstance(v, float) and v != v)) else float(v)
        out.append({"x": str(t), "y": y})
    return out


# ── pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/insurance")
def insurance():
    return render_template("insurance.html")


# ── API: data management ──────────────────────────────────────────────────────

@app.route("/api/data/load_defaults", methods=["POST"])
def api_load_defaults():
    """Load default workspace data files."""
    _state["data_dir"] = str(ROOT_DIR)
    _state["uploaded_files"] = {}
    _state["pipeline"] = None
    results = {}
    for tid in TARGET_TURBINES + REFERENCE_TURBINES:
        p = ROOT_DIR / f"{tid}_data.xlsx"
        ok = p.exists()
        results[tid] = {"found": ok, "path": str(p) if ok else None}
    for tid in TARGET_TURBINES:
        p = ROOT_DIR / f"{tid}_power_curve.xlsx"
        ok = p.exists()
        results[f"{tid}_curve"] = {"found": ok}
    return jsonify({"status": "ok", "files": results})


@app.route("/api/data/upload", methods=["POST"])
def api_upload():
    """Accept uploaded xlsx, xls, or csv files."""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    fname = f.filename
    ext = Path(fname).suffix.lower()
    if ext not in [".xlsx", ".xls", ".csv"]:
        return jsonify({
            "status": "error",
            "message": f"Unsupported file extension '{ext}'. Supported formats: .xlsx, .xls, .csv"
        }), 400

    # Determine turbine_id from filename
    tid = None
    for t in TARGET_TURBINES + REFERENCE_TURBINES:
        if fname.startswith(t):
            tid = t
            break
    if tid is None:
        return jsonify({"status": "error",
                        "message": f"Cannot determine turbine from filename: {fname}. File name must start with turbine ID (e.g. T1_data.csv, T1_power_curve.xlsx)"}), 400

    save_path = APP_DIR / "uploads" / fname
    save_path.parent.mkdir(exist_ok=True)
    f.save(save_path)

    # Validate
    try:
        if "power_curve" in fname:
            df = load_power_curve(tid, file_obj=str(save_path))
            _state["uploaded_files"][f"{tid}_curve"] = str(save_path)
        else:
            df = load_turbine_data(tid, file_obj=str(save_path))
            _state["uploaded_files"][tid] = str(save_path)
        return jsonify({"status": "ok", "turbine": tid, "rows": len(df), "filename": fname, "format": ext})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ── API: outage detection & management ────────────────────────────────────────

@app.route("/api/outage/detect")
def api_outage_detect():
    """Detect candidate outage periods from target turbine data without running heavy models."""
    target_id = request.args.get("target_id", _state["target_id"])
    try:
        df = load_turbine_data(target_id, data_dir=_state["data_dir"])
        blocks = detect_null_blocks(df)
        formatted = []
        for b in blocks:
            formatted.append({
                "start": str(b["start"]),
                "end": str(b["end"]),
                "n_intervals": b["n_intervals"],
                "duration_hours": b["duration_hours"],
                "duration_str": b["duration_str"],
                "category": b["category"],
                "confidence": b["confidence"],
                "rationale": b["rationale"],
                "is_outage_candidate": b["is_outage_candidate"],
                "has_test_flag": b.get("has_test_flag", False),
            })
        return jsonify({
            "status": "ok",
            "target_id": target_id,
            "total_rows": len(df),
            "total_nulls": int(df["wind_speed"].isna().sum()),
            "candidates": formatted,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/outage/validate", methods=["POST"])
def api_outage_validate():
    """Validate a candidate or manual outage period against the target turbine dataset."""
    data = request.get_json(silent=True) or {}
    target_id = data.get("target_id", _state["target_id"])
    start = data.get("start")
    end = data.get("end")
    if not start or not end:
        return jsonify({"status": "error", "message": "Start and end timestamps are required."}), 400
    try:
        df = load_turbine_data(target_id, data_dir=_state["data_dir"])
        val_res = validate_outage_interval(df, start, end)
        return jsonify({"status": "ok", "target_id": target_id, "result": val_res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/outage/confirm", methods=["POST"])
def api_outage_confirm():
    """Confirm the chosen outage period (from automatic detection or manual entry)."""
    data = request.get_json(silent=True) or {}
    target_id = data.get("target_id", _state["target_id"])
    start = data.get("start")
    end = data.get("end")
    notes = data.get("notes", "")

    if not start or not end:
        return jsonify({"status": "error", "message": "Start and end timestamps are required."}), 400

    new_start = str(pd.Timestamp(start))
    new_end   = str(pd.Timestamp(end))

    # Determine if configuration changed relative to current pipeline run
    is_changed = (
        target_id != _state["target_id"]
        or new_start != _state["outage_start"]
        or new_end != _state["outage_end"]
    )

    _state["target_id"] = target_id
    _state["outage_start"] = new_start
    _state["outage_end"] = new_end
    _state["outage_notes"] = notes
    _state["outage_confirmed"] = True

    if is_changed and _state.get("pipeline") is not None:
        _state["pipeline_stale"] = True

    return jsonify({
        "status": "ok",
        "target_id": _state["target_id"],
        "outage_start": _state["outage_start"],
        "outage_end": _state["outage_end"],
        "outage_notes": _state["outage_notes"],
        "outage_confirmed": True,
        "is_stale": _state.get("pipeline_stale", False),
    })


@app.route("/api/diagnostics")
def api_diagnostics():
    """Return error diagnostics and energy deviations."""
    pipe = _state.get("pipeline")
    if pipe is None:
        return jsonify({"status": "not_run"}), 404

    diag = getattr(pipe, "diagnostics", {})
    base = getattr(pipe, "baseline_metrics", {})
    stm = pipe.shown_test_metrics

    # Sensitivity analysis documentation
    sensitivity_notes = (
        "Power Curve Nonlinearity: Between 4 and 11 m/s (the cubic rise and knee region), "
        "the power curve slope dP/dv ranges from 300 to 700+ kW/(m/s). In this interval, "
        "a 1.0 m/s wind prediction error translates to a large 300-700 kW power discrepancy. "
        "Conversely, above 12 m/s on the rated power plateau (4,500 kW), dP/dv is 0 kW/(m/s), "
        "meaning wind speed errors produce zero power error."
    )

    return jsonify({
        "status": "ok",
        "target_id": pipe.target_id,
        "selected_model": pipe.selected_model_name,
        "diagnostics": diag,
        "baseline_metrics": {k: _safe_float(v) for k, v in base.items()},
        "shown_test_metrics": {k: _safe_float(v) for k, v in stm.items()},
        "sensitivity_notes": sensitivity_notes,
    })


# ── API: configuration ────────────────────────────────────────────────────────

@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if "target_id" in data:
            _state["target_id"] = data["target_id"]
            _state["pipeline"] = None
        if "outage_start" in data:
            _state["outage_start"] = data["outage_start"]
            _state["pipeline"] = None
        if "outage_end" in data:
            _state["outage_end"] = data["outage_end"]
            _state["pipeline"] = None
        if "assume_zero" in data:
            _state["assume_zero"] = bool(data["assume_zero"])
    return jsonify({
        "target_id":    _state["target_id"],
        "outage_start": _state["outage_start"],
        "outage_end":   _state["outage_end"],
        "assume_zero":  _state["assume_zero"],
        "target_turbines":    TARGET_TURBINES,
        "reference_turbines": REFERENCE_TURBINES,
    })


# ── API: run pipeline ─────────────────────────────────────────────────────────

@app.route("/api/run", methods=["POST"])
def api_run():
    """Run the full analysis pipeline and cache results."""
    data = request.get_json(silent=True) or {}
    target_id    = data.get("target_id",    _state["target_id"])
    outage_start = pd.Timestamp(data.get("outage_start", _state["outage_start"]))
    outage_end   = pd.Timestamp(data.get("outage_end",   _state["outage_end"]))
    assume_zero  = data.get("assume_zero", _state["assume_zero"])

    _state["target_id"]    = target_id
    _state["outage_start"] = str(outage_start)
    _state["outage_end"]   = str(outage_end)
    _state["assume_zero"]  = assume_zero

    try:
        pipe = OutageAnalysisPipeline(
            target_id=target_id,
            outage_start=outage_start,
            outage_end=outage_end,
            data_dir=_state["data_dir"],
            assume_zero_production=assume_zero,
        )
        pipe.run()
        _state["pipeline"] = pipe
        _state["pipeline_stale"] = False
        return jsonify({"status": "ok", "summary": _safe_summary(pipe)})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback": tb}), 500


def _safe_summary(pipe: OutageAnalysisPipeline) -> dict:
    """Convert pipeline summary to JSON-safe dict."""
    s = pipe.to_summary_dict()
    for k, v in s.items():
        if isinstance(v, float) and v != v:
            s[k] = None
    return s


# ── API: outage detection ─────────────────────────────────────────────────────

@app.route("/api/outage_detection")
def api_outage_detection():
    """Return observed + predicted wind-speed series for the outage page."""
    pipe = _state.get("pipeline")
    if pipe is None:
        return jsonify({"status": "not_run"}), 404

    target_df = load_turbine_data(pipe.target_id, data_dir=_state["data_dir"])

    # Window: 14 days before outage start → outage end + 3 days
    window_start = pipe.outage_start - pd.Timedelta(days=14)
    window_end   = pipe.outage_end   + pd.Timedelta(days=3)
    window_df    = target_df[
        (target_df["timestamp"] >= window_start)
        & (target_df["timestamp"] <= window_end)
    ]

    observed = _series_to_json(
        window_df["timestamp"],
        window_df["wind_speed"],
    )

    predicted = []
    if pipe.outage_df is not None:
        predicted = _series_to_json(
            pipe.outage_df["timestamp"],
            pipe.outage_df["predicted_wind_speed"],
        )

    null_blocks = detect_null_blocks(target_df)
    blocks_out = [
        {
            "start": str(b["start"]),
            "end":   str(b["end"]),
            "n":     b["n_intervals"],
            "hours": round(b["duration_hours"], 2),
            "is_outage": b["is_outage_candidate"],
        }
        for b in null_blocks
    ]

    return jsonify({
        "status":          "ok",
        "observed":        observed,
        "predicted":       predicted,
        "null_blocks":     blocks_out,
        "outage_start":    str(pipe.outage_start),
        "outage_end":      str(pipe.outage_end),
        "n_missing":       int(target_df["wind_speed"].isna().sum()),
        "n_outage_intervals": pipe.n_outage_intervals,
        "selected_model":  pipe.selected_model_name,
    })


@app.route("/api/outage_detection/download_csv")
def api_outage_csv():
    """Download predicted outage interval values as CSV."""
    pipe = _state.get("pipeline")
    if pipe is None or pipe.outage_df is None:
        abort(404)

    df = pipe.outage_df[
        ["timestamp", "predicted_wind_speed", "predicted_power_kw", "energy_kwh", "cumulative_energy_kwh"]
    ].copy()
    df.columns = [
        "timestamp", "predicted_wind_speed_ms",
        "predicted_power_kw", "energy_kwh_per_interval", "cumulative_energy_kwh"
    ]

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=outage_predictions.csv"},
    )


# ── API: explainability ───────────────────────────────────────────────────────

@app.route("/api/explainability")
def api_explainability():
    pipe = _state.get("pipeline")
    if pipe is None:
        return jsonify({"status": "not_run"}), 404

    corr = pipe.corr_df.copy()
    corr_list = corr.to_dict(orient="records")
    for row in corr_list:
        for k in ["pearson", "spearman"]:
            row[k] = _safe_float(row[k])

    model = pipe.selected_model
    features = getattr(model, "feature_names", []) if model else []

    return jsonify({
        "status":              "ok",
        "correlations":        corr_list,
        "selected_model":      pipe.selected_model_name,
        "ref_turbines":        pipe.ref_turbines,
        "features":            features,
        "val_start":           str(pipe.val_start) if pipe.val_start else None,
        "val_end":             str(pipe.val_end)   if pipe.val_end   else None,
        "n_val":               pipe.n_val,
        "val_mae":             _safe_float(pipe.val_metrics.get("mae")),
        "val_rmse":            _safe_float(pipe.val_metrics.get("rmse")),
        "val_wmape":           _safe_float(pipe.val_metrics.get("wmape")),
        "outage_start":        str(pipe.outage_start),
    })


# ── API: energy loss ──────────────────────────────────────────────────────────

@app.route("/api/energy_loss")
def api_energy_loss():
    pipe = _state.get("pipeline")
    if pipe is None:
        return jsonify({"status": "not_run"}), 404

    if pipe.outage_df is None:
        return jsonify({"status": "no_outage_data"}), 200

    df = pipe.outage_df

    power_series = _series_to_json(df["timestamp"], df["predicted_power_kw"])
    cum_energy   = _series_to_json(df["timestamp"], df["cumulative_energy_kwh"])

    max_power = float(pipe.power_curve["power"].max())

    avg_power = _safe_float(df["predicted_power_kw"].mean())
    total_kwh = _safe_float(pipe.total_energy_kwh)

    return jsonify({
        "status":             "ok",
        "power_series":       power_series,
        "cum_energy_series":  cum_energy,
        "total_energy_kwh":   total_kwh,
        "avg_power_kw":       avg_power,
        "max_rated_power_kw": max_power,
        "outage_start":       str(pipe.outage_start),
        "outage_end":         str(pipe.outage_end),
        "duration_str":       pipe.outage_duration_str,
        "n_intervals":        pipe.n_outage_intervals,
        "assume_zero":        pipe.assume_zero_production,
    })


# ── API: metrics ──────────────────────────────────────────────────────────────

@app.route("/api/metrics")
def api_metrics():
    pipe = _state.get("pipeline")
    if pipe is None:
        return jsonify({"status": "not_run"}), 404

    all_names = ["LightGBM", "Random Forest", "XGBoost", "ARIMA/SARIMA"]
    # Ensure any additional model in metrics is included
    for k in pipe.model_metrics:
        if k not in all_names:
            all_names.append(k)

    model_table = []
    for name in all_names:
        if name not in pipe.model_metrics:
            continue
        m   = pipe.model_metrics.get(name, {})
        res = pipe.model_results.get(name)
        rank_pos = None
        if name in pipe.ranked_models:
            rank_pos = pipe.ranked_models.index(name) + 1
        model_table.append({
            "name":        name,
            "available":   m.get("available", False),
            "error_msg":   getattr(res, "error_msg", None),
            "n_refs":      m.get("n_refs", 0),
            "ref_set":     ", ".join(getattr(res, "reference_turbines", [])) or "Univariate / Mean",
            "n_val":       m.get("n", 0),
            "mae":         _safe_float(m.get("mae")),
            "rmse":        _safe_float(m.get("rmse")),
            "wmape":       _safe_float(m.get("wmape")),
            "rank":        rank_pos,
            "selected":    name == pipe.selected_model_name,
        })

    # Shown-test series
    st_observed  = []
    st_predicted = []
    if pipe.shown_test_df is not None:
        st_observed  = _series_to_json(
            pipe.shown_test_df["timestamp"], pipe.shown_test_df["actual_wind_speed"]
        )
        st_predicted = _series_to_json(
            pipe.shown_test_df["timestamp"], pipe.shown_test_df["predicted_wind_speed"]
        )

    st_m = pipe.shown_test_metrics
    b_comp = getattr(pipe, "baseline_comparisons", {})
    return jsonify({
        "status":           "ok",
        "model_table":      model_table,
        "selected_model":   pipe.selected_model_name,
        "val_metrics":      {k: _safe_float(v) for k, v in pipe.val_metrics.items()},
        "baseline_comparisons": {
            k: {mk: _safe_float(mv) for mk, mv in v.items()}
            for k, v in b_comp.items()
        },
        "shown_test_metrics": {
            "n":                    st_m.get("n", 0),
            "mae":                  _safe_float(st_m.get("mae")),
            "rmse":                 _safe_float(st_m.get("rmse")),
            "wmape":                _safe_float(st_m.get("wmape")),
            "bias":                 _safe_float(st_m.get("bias")),
            "power_mae":            _safe_float(st_m.get("power_mae")),
            "power_rmse":           _safe_float(st_m.get("power_rmse")),
            "energy_deviation_pct": _safe_float(st_m.get("energy_deviation_pct")),
        },
        "shown_test_start":  str(SHOWN_TEST_START),
        "shown_test_end":    str(SHOWN_TEST_END),
        "st_observed":       st_observed,
        "st_predicted":      st_predicted,
        "val_start":         str(pipe.val_start) if pipe.val_start else None,
        "val_end":           str(pipe.val_end)   if pipe.val_end   else None,
        "n_val":             pipe.n_val,
    })


@app.route("/api/metrics/shown_test_csv")
def api_shown_test_csv():
    """Download shown-test predictions vs actuals as CSV."""
    pipe = _state.get("pipeline")
    if pipe is None or pipe.shown_test_df is None:
        abort(404)

    df = pipe.shown_test_df[
        ["timestamp", "actual_wind_speed", "predicted_wind_speed",
         "actual_power_kw", "predicted_power_kw"]
    ].copy()

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=shown_test_predictions.csv"},
    )


# ── API: overview ─────────────────────────────────────────────────────────────

@app.route("/api/overview")
def api_overview():
    """Return KPI cards and active configuration for Overview."""
    pipe = _state.get("pipeline")
    is_stale = _state.get("pipeline_stale", False)

    active_config = {
        "target_id": _state["target_id"],
        "outage_start": _state["outage_start"],
        "outage_end": _state["outage_end"],
        "outage_notes": _state.get("outage_notes", ""),
        "outage_confirmed": _state.get("outage_confirmed", True),
        "is_stale": is_stale,
        "assume_zero": _state.get("assume_zero", True),
    }

    if pipe is None:
        return jsonify({
            "status": "not_run",
            "active_config": active_config,
            "turbines": [],
        }), 200

    # For selected turbine, return cached pipeline result
    cards = []
    t = pipe.target_id
    s = pipe.to_summary_dict()
    cards.append({
        "turbine":        t,
        "selected_model": s["selected_model"],
        "val_mae":        _safe_float(s["val_mae"]),
        "val_rmse":       _safe_float(s["val_rmse"]),
        "val_wmape":      _safe_float(s["val_wmape"]),
        "val_bias":       _safe_float(s.get("val_bias")),
        "baseline_val_mae": _safe_float(s.get("baseline_val_mae")),
        "baseline_val_wmape": _safe_float(s.get("baseline_val_wmape")),
        "energy_kwh":     _safe_float(s["total_energy_kwh"]),
        "outage_start":   s["outage_start"],
        "outage_end":     s["outage_end"],
        "outage_duration_str": s.get("outage_duration_str", ""),
        "assume_zero":    s.get("assume_zero_production", True),
        "is_selected":    True,
        "is_stale":       is_stale,
        "warnings":       s["status_messages"],
    })

    return jsonify({
        "status": "ok",
        "active_config": active_config,
        "turbines": cards,
    })


# ── API: advanced panel ───────────────────────────────────────────────────────

@app.route("/api/advanced")
def api_advanced():
    """Return detailed technical information for the Advanced panel."""
    pipe = _state.get("pipeline")
    if pipe is None:
        return jsonify({"status": "not_run"}), 404

    corr = pipe.corr_df.copy()
    corr_records = []
    for _, row in corr.iterrows():
        corr_records.append({
            "turbine":  row["turbine"],
            "pearson":  _safe_float(row["pearson"]),
            "spearman": _safe_float(row["spearman"]),
            "n_paired": int(row["n_paired"]),
        })

    all_names = ["LightGBM", "Random Forest", "XGBoost", "ARIMA/SARIMA"]
    for k in pipe.model_metrics:
        if k not in all_names:
            all_names.append(k)

    model_details = []
    for name in all_names:
        if name not in pipe.model_metrics:
            continue
        m   = pipe.model_metrics.get(name, {})
        res = pipe.model_results.get(name)
        model_details.append({
            "name":        name,
            "available":   m.get("available", False),
            "error_msg":   getattr(res, "error_msg", None),
            "n_train":     getattr(res, "n_train", 0),
            "n_val":       m.get("n", 0),
            "mae":         _safe_float(m.get("mae")),
            "rmse":        _safe_float(m.get("rmse")),
            "wmape":       _safe_float(m.get("wmape")),
            "ref_turbines": getattr(res, "reference_turbines", []),
            "features":    getattr(res, "feature_names", []),
            "rank":        (pipe.ranked_models.index(name) + 1) if name in pipe.ranked_models else None,
            "selected":    name == pipe.selected_model_name,
        })

    return jsonify({
        "status":         "ok",
        "correlations":   corr_records,
        "model_details":  model_details,
        "ranked_models":  pipe.ranked_models,
        "selection_rule": (
            "Deterministic ranking across wind-speed MAE, RMSE, and WMAPE on chronological validation holdout. "
            "LightGBM, Random Forest, XGBoost, and ARIMA are objectively compared on unseen pre-outage validation data."
        ),
        "val_window": {
            "start": str(pipe.val_start) if pipe.val_start else None,
            "end":   str(pipe.val_end)   if pipe.val_end   else None,
            "n":     pipe.n_val,
        },
        "data_cutoff": str(pipe.outage_start),
        "random_seed": 42,
        "lags_used":   [1, 2, 3],
        "lag_description": "10-, 20-, 30-minute past reference wind speeds",
        "limitations": [
            "Reference turbines from different parks may have different local wind regimes.",
            "ARIMA/SARIMA is univariate and uses only the target's own history.",
            "Tree models use past-only lagged reference measurements; no future values.",
            "Power conversion uses linear interpolation within the curve; clamped at boundary.",
            "Energy estimate assumes zero production during outage unless checkbox is unchecked.",
            "Uncertainty is not calibrated; no prediction intervals are provided.",
            "Wind direction, temperature, and pressure are not used as model features.",
        ],
    })


# ── API: insurance report ─────────────────────────────────────────────────────

@app.route("/api/insurance_report")
def api_insurance_report():
    """Return all data needed for the insurance report page."""
    pipe = _state.get("pipeline")
    if pipe is None:
        return jsonify({"status": "not_run"}), 404

    st_m = pipe.shown_test_metrics
    out_df = pipe.outage_df

    # Build interval table for the report
    interval_rows = []
    if out_df is not None:
        for _, row in out_df.iterrows():
            interval_rows.append({
                "timestamp":       str(row["timestamp"]),
                "predicted_wind":  _safe_float(row["predicted_wind_speed"]),
                "predicted_power": _safe_float(row["predicted_power_kw"]),
                "energy_kwh":      _safe_float(row["energy_kwh"]),
                "cum_energy_kwh":  _safe_float(row["cumulative_energy_kwh"]),
            })

    corr_top3 = pipe.corr_df.head(3).to_dict(orient="records")
    for row in corr_top3:
        for k in ["pearson", "spearman"]:
            row[k] = _safe_float(row[k])

    return jsonify({
        "status":              "ok",
        "target_id":           pipe.target_id,
        "outage_start":        str(pipe.outage_start),
        "outage_end":          str(pipe.outage_end),
        "duration_str":        pipe.outage_duration_str,
        "n_intervals":         pipe.n_outage_intervals,
        "selected_model":      pipe.selected_model_name,
        "ref_turbines":        pipe.ref_turbines,
        "val_metrics":         {k: _safe_float(v) for k, v in pipe.val_metrics.items()},
        "shown_test_metrics":  {k: _safe_float(v) for k, v in st_m.items()},
        "total_energy_kwh":    _safe_float(pipe.total_energy_kwh),
        "assume_zero":         pipe.assume_zero_production,
        "corr_top3":           corr_top3,
        "interval_rows":       interval_rows,
        "warnings":            pipe.status_messages,
        "power_curve_max_kw":  float(pipe.power_curve["power"].max()),
        "power_curve_max_ws":  float(pipe.power_curve["wind_speed"].max()),
    })


@app.route("/api/insurance_report/download_csv")
def api_insurance_csv():
    """Download detailed insurance report interval CSV."""
    pipe = _state.get("pipeline")
    if pipe is None or pipe.outage_df is None:
        abort(404)

    df = pipe.outage_df[[
        "timestamp", "predicted_wind_speed", "predicted_power_kw",
        "energy_kwh", "cumulative_energy_kwh"
    ]].copy()
    df.columns = [
        "timestamp", "predicted_wind_speed_ms",
        "predicted_power_kw", "energy_kwh_per_interval", "cumulative_energy_kwh"
    ]

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=insurance_report_intervals.csv"},
    )


@app.route("/api/insurance_report/download_html")
def api_insurance_html():
    """Generate and download a standalone HTML insurance report."""
    pipe = _state.get("pipeline")
    if pipe is None:
        abort(404)

    st_m = pipe.shown_test_metrics
    out_df = pipe.outage_df

    def fmt(v, d=3):
        if v is None or (isinstance(v, float) and v != v):
            return "N/A"
        return f"{v:,.{d}f}"

    def fmt_kwh(v):
        if v is None or (isinstance(v, float) and v != v):
            return "N/A"
        return f"{v:,.1f} kWh"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Wind Farm Outage Insurance Report – {pipe.target_id}</title>
<style>
  body {{font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #1a1a1a; line-height: 1.6;}}
  h1 {{color: #1a5c38; border-bottom: 3px solid #1a5c38; padding-bottom: 8px;}}
  h2 {{color: #1a5c38; margin-top: 32px;}}
  table {{border-collapse: collapse; width: 100%; margin: 16px 0;}}
  th {{background: #1a5c38; color: white; padding: 8px 12px; text-align: left;}}
  td {{padding: 7px 12px; border-bottom: 1px solid #ddd;}}
  tr:nth-child(even) {{background: #f7fdf9;}}
  .kpi {{display: inline-block; background: #f7fdf9; border: 1px solid #b2dfca;
         border-radius: 8px; padding: 16px 24px; margin: 8px; min-width: 160px;}}
  .kpi-label {{font-size: 12px; color: #555; text-transform: uppercase;}}
  .kpi-value {{font-size: 24px; font-weight: 700; color: #1a5c38;}}
  .warning {{background: #fff8e1; border-left: 4px solid #f9a825; padding: 10px 16px; margin: 12px 0;}}
  .disclaimer {{background: #fce4ec; border-left: 4px solid #c62828; padding: 10px 16px; margin: 20px 0; font-size: 13px;}}
  .footer {{margin-top: 48px; font-size: 12px; color: #888; border-top: 1px solid #ddd; padding-top: 12px;}}
</style>
</head>
<body>
<h1>🌬️ Wind Farm Outage Analysis – Insurance Report</h1>
<p><strong>Target Turbine:</strong> {pipe.target_id} &nbsp;|&nbsp;
   <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC</p>

<div class="disclaimer">
⚠️ <strong>Important:</strong> The energy figure below represents <em>expected normal energy</em>
estimated from predicted wind speeds and the supplied power curve.
It is a potential loss estimate assuming zero production during the outage,
<strong>not a confirmed lost-energy amount or verified insurance claim.</strong>
Actual generation depends on turbine availability, grid conditions, curtailment,
and other factors not captured here.
</div>

<h2>1. Outage Period</h2>
<div>
  <span class="kpi"><div class="kpi-label">Start</div><div class="kpi-value" style="font-size:16px">{pipe.outage_start.strftime('%Y-%m-%d %H:%M')}</div></span>
  <span class="kpi"><div class="kpi-label">End</div><div class="kpi-value" style="font-size:16px">{pipe.outage_end.strftime('%Y-%m-%d %H:%M')}</div></span>
  <span class="kpi"><div class="kpi-label">Duration</div><div class="kpi-value" style="font-size:16px">{pipe.outage_duration_str}</div></span>
  <span class="kpi"><div class="kpi-label">Intervals</div><div class="kpi-value" style="font-size:16px">{pipe.n_outage_intervals}</div></span>
</div>

<h2>2. Selected Model</h2>
<table>
<tr><th>Model</th><th>Reference Turbines</th><th>Val MAE (m/s)</th><th>Val RMSE (m/s)</th><th>Val WMAPE (%)</th><th>Val n</th></tr>
<tr>
  <td><strong>{pipe.selected_model_name or 'N/A'}</strong></td>
  <td>{', '.join(pipe.ref_turbines) or 'N/A'}</td>
  <td>{fmt(pipe.val_metrics.get('mae'))}</td>
  <td>{fmt(pipe.val_metrics.get('rmse'))}</td>
  <td>{fmt(pipe.val_metrics.get('wmape'))}</td>
  <td>{pipe.val_metrics.get('n', 'N/A')}</td>
</tr>
</table>

<h2>3. Shown-Test Performance (Sep 4–9, observed actuals)</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Wind Speed MAE (m/s)</td><td>{fmt(st_m.get('mae'))}</td></tr>
<tr><td>Wind Speed RMSE (m/s)</td><td>{fmt(st_m.get('rmse'))}</td></tr>
<tr><td>Wind Speed WMAPE (%)</td><td>{fmt(st_m.get('wmape'))}</td></tr>
<tr><td>Power MAE (kW)</td><td>{fmt(st_m.get('power_mae'), 1)}</td></tr>
<tr><td>Power RMSE (kW)</td><td>{fmt(st_m.get('power_rmse'), 1)}</td></tr>
<tr><td>Energy Deviation (%)</td><td>{fmt(st_m.get('energy_deviation_pct'), 2)}</td></tr>
<tr><td>Valid Intervals</td><td>{st_m.get('n', 'N/A')}</td></tr>
</table>
<p><em>Note: Shown-test results use observed Sep 4–9 actuals, separate from validation.
These results were not used to select the model.</em></p>

<h2>4. Estimated Energy (Potential Loss)</h2>
<div>
  <span class="kpi"><div class="kpi-label">Total Expected Energy</div><div class="kpi-value">{fmt_kwh(pipe.total_energy_kwh)}</div></span>
  <span class="kpi"><div class="kpi-label">Intervals</div><div class="kpi-value">{pipe.n_outage_intervals}</div></span>
</div>
<p>
  Energy computed as: predicted power (kW) × 1/6 hour per 10-minute interval, summed over the outage.
  Power derived from predicted wind speed via linear interpolation of the supplied power curve.
  Clamped at curve boundaries; not extrapolated above {float(pipe.power_curve['wind_speed'].max()):.1f} m/s.
</p>

<h2>5. Reference Turbine Correlations (pre-outage, paired observations)</h2>
<table>
<tr><th>Turbine</th><th>Pearson r</th><th>Spearman r</th><th>Paired n</th></tr>
{"".join(f"<tr><td>{r['turbine']}</td><td>{fmt(r.get('pearson'),3)}</td><td>{fmt(r.get('spearman'),3)}</td><td>{int(r.get('n_paired',0))}</td></tr>" for _, r in pipe.corr_df.iterrows())}
</table>
<p><em>Correlation measures association, not predictive accuracy. Computed on pre-outage data only.</em></p>

<h2>6. Key Assumptions and Limitations</h2>
<ul>
  <li>Zero production during the outage is assumed to label expected energy as potential loss.</li>
  <li>Prediction uncertainty is not calibrated; no prediction intervals are provided.</li>
  <li>Reference turbines are from different wind parks and may have different local wind regimes.</li>
  <li>Power conversion uses linear interpolation within the curve; extrapolation is disabled.</li>
  <li>Actual grid delivery, curtailment, and turbine-specific availability are not captured.</li>
  <li>Training data covers {pipe.outage_start.strftime('%Y-%m-%d')} (exclusive) back to 2026-07-01.</li>
  <li>Validation period: {pipe.val_start} to {pipe.val_end} ({pipe.n_val} intervals).</li>
</ul>

<div class="footer">
  Generated by BayWa r.e. Wind Farm Outage Analysis Tool.
  For hackathon demonstration purposes only. Not a legal or actuarial document.
</div>
</body></html>"""

    buf = io.BytesIO(html.encode("utf-8"))
    return send_file(
        buf,
        mimetype="text/html",
        as_attachment=True,
        download_name=f"insurance_report_{pipe.target_id}.html",
    )


# ── health check ──────────────────────────────────────────────────────────────

@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BayWa r.e. Wind Farm Outage Analysis")
    print("Open: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
