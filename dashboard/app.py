"""Read-only Streamlit dashboard for railway-route reliability."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RailPulse | Railway Reliability", layout="wide")
st.title("RailPulse — Railway Reliability Intelligence")
st.caption("Independent ONCF-inspired portfolio project · Open SNCF Intercités data, NOT ONCF data.")
base = Path(__file__).resolve().parents[1] / "artifacts"
if not (base / "metrics.json").exists():
    st.info("First run: railpulse fetch && railpulse train")
    st.stop()
metrics = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
history = pd.read_csv(base / "backtest.csv", parse_dates=["date"])
future = pd.read_csv(base / "forecast_next_month.csv")
a, b, c = st.columns(3)
a.metric("ML MAE (percentage points)", metrics["random_forest_mae_percentage_points"])
b.metric("Lag-1 baseline MAE", metrics["baseline_mae_percentage_points"])
c.metric("Routes in test sample", metrics["test_routes"])
route = st.selectbox("Select origin–destination route", sorted(history["route"].unique()))
st.line_chart(history[history["route"] == route].set_index("date")[["late_rate", "predicted_late_rate", "baseline_lag1"]])
st.subheader("Next-month route-level predictions")
if len(future):
    st.dataframe(future.sort_values("predicted_late_rate", ascending=False), hide_index=True, use_container_width=True)
else:
    st.warning("Insufficient recent contiguous records for future forecasts.")
st.caption("Target is a MONTHLY share of delayed train arrivals at the last station. Not a per-train lateness probability; not comparable to ONCF metrics.")
