"""Chronological holdout and baseline-aware ML evaluation."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .features import FEATURES, NUMERIC_FEATURES, make_features, make_next_month_features

def temporal_split(df, test_months=6):
    months = sorted(df["date"].unique())
    if len(months) < test_months + 12:
        raise ValueError("At least 12 training and 6 testing months required")
    cutoff = months[-test_months]
    train, test = df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy()
    return train, test

def new_model():
    transform = ColumnTransformer([
        ("route", OneHotEncoder(handle_unknown="ignore"), ["route"]),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("features", transform),
        ("forest", RandomForestRegressor(n_estimators=160, min_samples_leaf=3, random_state=42, n_jobs=1)),
    ])

def train_and_evaluate(operational, output="artifacts"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = make_features(operational)
    train, test = temporal_split(frame)
    model = new_model().fit(train[FEATURES], train["late_rate"])
    actual = test["late_rate"].to_numpy()
    predicted = np.clip(model.predict(test[FEATURES]), 0, 100)
    baseline = test["late_rate_lag_1"].to_numpy()
    metrics = {
        "source": "SNCF Intercites open data (NOT ONCF)",
        "target": "monthly late-train percentage per route",
        "train_end": str(train["date"].max().date()),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "train_records": int(len(train)),
        "test_records": int(len(test)),
        "test_routes": int(test["route"].nunique()),
        "baseline_mae_percentage_points": round(float(mean_absolute_error(actual, baseline)), 3),
        "random_forest_mae_percentage_points": round(float(mean_absolute_error(actual, predicted)), 3),
        "random_forest_rmse_percentage_points": round(float(np.sqrt(mean_squared_error(actual, predicted))), 3),
    }
    backtest = test[["date", "route", "late_rate", "operated"]].copy()
    backtest["baseline_lag1"] = baseline
    backtest["predicted_late_rate"] = predicted
    backtest.to_csv(output / "backtest.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    final_model = new_model().fit(frame[FEATURES], frame["late_rate"])
    joblib.dump(final_model, output / "model.joblib")
    future = make_next_month_features(operational)
    if len(future):
        future["predicted_late_rate"] = np.clip(final_model.predict(future[FEATURES]), 0, 100)
    future[["date", "route", "predicted_late_rate"]].to_csv(output / "forecast_next_month.csv", index=False) if len(future) else (output / "forecast_next_month.csv").write_text("date,route,predicted_late_rate\n")
    return metrics
