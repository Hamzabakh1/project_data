"""Synthetic fixtures are ONLY unit tests, not research findings."""
import numpy as np
import pandas as pd
import pytest
from railpulse.prepare import load_and_validate
from railpulse.features import FEATURES, make_features, make_next_month_features
from railpulse.model import temporal_split, train_and_evaluate

def synthetic_csv(tmp_path):
    records = []
    for r in range(3):
        for m, date in enumerate(pd.date_range("2021-01-01", periods=44, freq="MS")):
            operated, cancelled = 100 + 7*r, m % 3
            records.append({
                "date": date.strftime("%Y-%m"),
                "depart": f"A{r}", "arrivee": f"B{r}",
                "nombre_de_trains_programmes": operated+cancelled,
                "nombre_de_trains_ayant_circule": operated,
                "nombre_de_trains_annules": cancelled,
                "nombre_de_trains_en_retard_a_l_arrivee": 8+m%6+2*r,
            })
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(records).to_csv(path, index=False, sep=";")
    return path

def test_validation_and_time_split(tmp_path):
    df = load_and_validate(synthetic_csv(tmp_path))
    assert len(df) == 132
    train, test = temporal_split(make_features(df))
    assert train.date.max() < test.date.min()
    assert test.date.nunique() == 6

def test_no_target_leakage(tmp_path):
    data = load_and_validate(synthetic_csv(tmp_path))
    original = make_features(data)
    day = pd.Timestamp("2023-01-01")
    mask = (data.date == day) & (data.route == "A0 → B0")
    data.loc[mask, "late_rate"] = 95
    changed = make_features(data)
    left = original.loc[(original.date == day) & (original.route == "A0 → B0"), FEATURES].reset_index(drop=True)
    right = changed.loc[(changed.date == day) & (changed.route == "A0 → B0"), FEATURES].reset_index(drop=True)
    assert left.equals(right)

def test_gap_in_history_prevents_fake_lag(tmp_path):
    df = load_and_validate(synthetic_csv(tmp_path))
    df = df[~((df.date == pd.Timestamp("2023-07-01")) & (df.route == "A0 → B0"))]
    frame = make_features(df)
    assert not ((frame.date == pd.Timestamp("2023-08-01")) & (frame.route == "A0 → B0")).any()

def test_bad_counts_rejected(tmp_path):
    path = synthetic_csv(tmp_path)
    df = pd.read_csv(path, sep=";")
    df.loc[0, "nombre_de_trains_en_retard_a_l_arrivee"] = 500
    df.to_csv(path, index=False, sep=";")
    with pytest.raises(ValueError, match="Delayed trains"):
        load_and_validate(path)

def test_full_pipeline(tmp_path):
    data = load_and_validate(synthetic_csv(tmp_path))
    assert len(make_next_month_features(data)) == 3
    report = train_and_evaluate(data, tmp_path / "out")
    assert report["test_records"] == 18
    assert np.isfinite(report["random_forest_mae_percentage_points"])
    for name in ("model.joblib", "metrics.json", "backtest.csv", "forecast_next_month.csv"):
        assert (tmp_path / "out" / name).exists()
