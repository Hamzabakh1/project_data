# RailPulse — Railway Reliability Intelligence

Independent Data Science portfolio project inspired by railway operational use cases. **Not affiliated with ONCF. No ONCF data used.**

Forecast monthly train-delay rates **by origin–destination rail route**, using publicly available **SNCF Intercités** monthly aggregates.

## Data
Official SNCF dataset: https://ressources.data.sncf.com/explore/dataset/regularite-mensuelle-intercites/ (ODbL licence). Source reporting is aggregated by route/month, not individual trains. SNCF's composite punctuality metric uses delay thresholds depending on journey length. Results are not ONCF performance measurements.

## Quick start
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev,dashboard]'
railpulse fetch --output data/regularite-mensuelle-intercites.csv
railpulse train --csv data/regularite-mensuelle-intercites.csv --output artifacts
streamlit run dashboard/app.py
pytest -q
```

## Workflow
Official CSV → data quality checks → monthly lag features → chronological 6-month holdout → Random Forest vs naive lag-1 baseline → model metrics → next-month route forecast → Streamlit dashboard.

## Methodological integrity
- Target: 100 × delayed arrivals / operated trains per route and month, **not train-level probability**.
- Feature creation uses only information known before forecast month; missing months are treated explicitly.
- Test months are strictly later than training months; no random train/test shuffle.
- ML results are benchmarked against the previous-month naive forecast.
- Test fixtures are synthetic, **not evidence of real-data performance**; download the official CSV to compute actual metrics.
- Geographic and operational transfer to ONCF is **not validated**.
- Respect source attribution and Open Database License (ODbL).

## Structure
`src/railpulse/` ingestion, validation, features, modeling, CLI; `dashboard/` Streamlit; `tests/` unit tests; `docs/` project brief; `.github/workflows/` CI.

**CV description:** RailPulse — Prévision de la régularité ferroviaire (projet personnel) · Python, Pandas, scikit-learn, Streamlit. Analyser la ponctualité ferroviaire sur données ouvertes et prévoir le taux mensuel de retard par liaison avec validation chronologique et comparaison à une référence naïve.
