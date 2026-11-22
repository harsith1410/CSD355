# Separating Driver Skill from Car Performance in Formula 1

CSD355 Foundation of Data Science project: a teammate-chain analysis of qualifying lap times, 2014 to round 17 of 2026.

Teammates drive the same car, so the qualifying gap between them is mostly a driver gap. We compute that gap for every session, summarise each pair and season with a 95% confidence interval, link drivers who were never teammates through shared teammates, estimate each car's pace with the driver effect removed, and use an ML model to predict next season's teammate gap.

## How to run

Run the notebooks in `src/` in this order (each one reads the files the earlier ones save):

| Step | Notebook | What it does | Main outputs |
|---|---|---|---|
| 1 | `build_raw.py` / `build_gaps.ipynb` | Extract F1DB qualifying data, convert lap times, Jolpica spot-check, pair teammates, compute gaps (Eq. 1) | `qualifying_clean.csv`, `data/teammate_gaps.csv` |
| 2 | `analysis.ipynb` | Exploratory analysis and outliers | figures |
| 3 | `pair_season_stats.ipynb` | IQR outliers (Eq. 2), t-intervals (Eq. 3), head-to-head, τ², pilot check | `data/pair_seasons.csv`, `figures/forest_2025_2026.png` |
| 4 | `ratings_car_pace.ipynb` | Driver ratings vs Russell, chain backtest, car pace | `data/driver_ratings.csv`, `data/car_pace.csv`, 3 figures |
| 5 | `ml_model.ipynb` | ML model with leakage-safe features, validation 2016–2024, test 2025–2026 | `data/ml_predictions.csv`, 2 figures |

Shared functions for steps 3–5 are in `src/f1_stats.py`.

```bash
pip install -r requirements.txt
pip install networkx scikit-learn     # needed for steps 4 and 5
```

## Key results

| Question | Result |
|---|---|
| Data | 5,465 qualifying entries extracted (5,448 after removing DSQ/EX), 269 race weekends, 63 drivers; 201 of 202 Jolpica rows match (the one gap is a row Jolpica does not list, not a timing difference) |
| Teammate comparisons | 2,655 sessions; 170 pair-seasons, 137 with ≥ 8 sessions |
| Outliers (1.5 × IQR) | 210 sessions (7.9%); skewness −0.84 → −0.21 after removal |
| Precision | Typical 95% CI half-width ±0.19% of lap time; 71 of 137 pair-seasons separate the two drivers |
| Season-to-season drift | τ² = 0.0153 (from 61 pairs seen in consecutive seasons) |
| Pilot (report Table VI) | Reproduced: Antonelli vs Russell 2025 +0.38% (95% CI +0.27 to +0.48), Russell 21–3 |
| Chain backtest | Picks the faster driver 71% of the time; 95% intervals cover 82% of real gaps |
| Car pace sanity check | Constructors' champion fastest or 2nd fastest in qualifying pace in 12 of 12 completed seasons |
| ML (test 2025–2026) | Hybrid model MAE 0.165 vs 0.198 (last season's gap) and 0.232 (guess 0); names the faster teammate 86% of the time |

## Data

F1DB, the open-source Formula 1 database (https://github.com/f1db/f1db), CC BY 4.0. Spot-checked against the Jolpica-F1 API (https://api.jolpi.ca/ergast/f1/).