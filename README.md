Case Study 1 — Turnover Outlier Detection


This repository implements a small end-to-end pipeline that:

- generates synthetic daily turnover data for 10 example stocks (60 days each),
- computes the required `chg_turnover` metric (mean of latest 30 days minus mean of previous 30 days),
- runs four outlier detectors (ABOD / KNN / IsolationForest / HBOS),
- produces per-model scores and an ensemble score, and
- provides an interactive Streamlit dashboard for visualization, thresholding, explainability and lightweight monitoring.

---


1. Generate synthetic data:

2. Run the outlier pipeline (creates `Case1_turnover_outliers/data/outlier_results.csv`):

3. Run the dashboard:


Notes about dashboard features

- **Ensemble percentile slider**: tune the score-based cutoff (default 80th percentile) used to mark high-risk stocks.
- **Per-model score histograms**: view normalized model scores and the ensemble distribution with a threshold marker.
- **Composition modes**: select `Voting (>=2)`, `Ensemble percentile`, or `Custom thresholds` (per-model sliders) in the sidebar to compose final flags.
- **Custom composition**: when `Custom thresholds` is selected you can set per-model normalized-score thresholds and combine flags with OR / AND / Weighted logic.
- **Hyperparameter sweep (quick)**: lightweight sweep button in the sidebar that tries a few `KNN n_neighbors` and `IForest contamination` combinations and writes results to `Case1_turnover_outliers/data/hyperparam_results.json`.
- **Explainability**: the dashboard shows per-feature z-scores for a selected stock and will attempt a SHAP explanation when `shap` is installed (optional).
- **Monitoring snapshot**: click `Save flags snapshot` to append a CSV row to `Case1_turnover_outliers/data/flags_history.csv` capturing the current flags and configuration.

Troubleshooting & environment notes

- If `pip install -r requirements.txt` fails on Windows (build errors), try the trimmed set first:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --prefer-binary -r requirements-min.txt
```

```

Files of interest

- `Case1_turnover_outliers/generate_data.py` — generates the synthetic turnover data (600 rows; two stocks have injected anomalies by design).
- `Case1_turnover_outliers/outliers_model.py` — computes `chg_turnover`, fits ABOD/KNN/IForest/HBOS, writes per-model labels & scores, normalizes scores (`*_score_norm`), computes `ensemble_score`, and saves `outlier_results.csv`.
- `Case1_turnover_outliers/dashboard_streamlit.py` — Streamlit dashboard with filters, plots, composition options, hyperparameter quick-sweep, explainability and monitoring.
- `Case1_turnover_outliers/data/outlier_results.csv` — output produced by `outliers_model.py` consumed by the dashboard.

- `requirements.txt` — full pinned list used for reproducible builds 

