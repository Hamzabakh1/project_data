"""Validate published monthly route aggregates and create the target."""
import numpy as np
import pandas as pd

COLUMNS = {
    "date": "date", "depart": "origin", "arrivee": "destination",
    "nombre_de_trains_programmes": "scheduled",
    "nombre_de_trains_ayant_circule": "operated",
    "nombre_de_trains_annules": "cancelled",
    "nombre_de_trains_en_retard_a_l_arrivee": "late",
}

def load_and_validate(path):
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    missing = sorted(set(COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected SNCF columns: {missing}")
    df = df.rename(columns=COLUMNS)[list(COLUMNS.values())].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if df["date"].isna().any():
        raise ValueError("Invalid source date")
    for col in ("origin", "destination"):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {col}")
        df[col] = df[col].astype(str).str.strip().str.upper()
    for col in ("scheduled", "operated", "cancelled", "late"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any() or not np.isfinite(df[col]).all() or (df[col] < 0).any() or (df[col] % 1 != 0).any():
            raise ValueError(f"Invalid train counts: {col}")
    if (df["late"] > df["operated"]).any():
        raise ValueError("Delayed trains cannot exceed operated")
    if (df["operated"] > df["scheduled"]).any() or (df["cancelled"] > df["scheduled"]).any():
        raise ValueError("Train count exceeds scheduled")
    df["route"] = df["origin"] + " → " + df["destination"]
    if df.duplicated(["route", "date"]).any():
        raise ValueError("Duplicate route/month grain")
    df = df[(df["operated"] > 0) & (df["scheduled"] > 0)].copy()
    df["late_rate"] = 100 * df["late"] / df["operated"]
    df["cancellation_rate"] = 100 * df["cancelled"] / df["scheduled"]
    return df.sort_values(["route", "date"]).reset_index(drop=True)
