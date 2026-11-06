"""Temporal features: months t-1 through t-3, never target month t."""
import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["late_rate_lag_1", "late_rate_lag_2", "late_rate_lag_3",
                    "cancellation_rate_lag_1", "operated_lag_1",
                    "late_rolling_mean_3", "month_sin", "month_cos"]
FEATURES = ["route", *NUMERIC_FEATURES]

def _route_history(group, future=False):
    g = group.sort_values("date").set_index("date")
    if future:
        g = g.reindex(pd.date_range(g.index.min(), g.index.max() + pd.DateOffset(months=1), freq="MS"))
    else:
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="MS"))
    g.index.name = "date"
    g["route"] = group["route"].iloc[0]
    for n in (1, 2, 3):
        g[f"late_rate_lag_{n}"] = g["late_rate"].shift(n)
    g["cancellation_rate_lag_1"] = g["cancellation_rate"].shift(1)
    g["operated_lag_1"] = g["operated"].shift(1)
    g["late_rolling_mean_3"] = g["late_rate"].shift(1).rolling(3, min_periods=3).mean()
    g["month_sin"] = np.sin(2 * np.pi * g.index.month / 12)
    g["month_cos"] = np.cos(2 * np.pi * g.index.month / 12)
    return g.reset_index()

def make_features(data):
    if data.empty:
        raise ValueError("Empty operational data")
    combined = pd.concat([_route_history(g) for _, g in data.groupby("route")], ignore_index=True)
    return combined.dropna(subset=[*FEATURES, "late_rate"]).sort_values(["date", "route"]).reset_index(drop=True)

def make_next_month_features(data):
    latest = data["date"].max()
    rows = []
    for route, g in data.groupby("route"):
        if g["date"].max() != latest:
            continue
        x = _route_history(g, future=True)
        y = x[x["date"] == latest + pd.DateOffset(months=1)].dropna(subset=FEATURES)
        if not y.empty:
            rows.append(y)
    return pd.concat(rows, ignore_index=True)[["date", *FEATURES]] if rows else pd.DataFrame(columns=["date", *FEATURES])
