from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from f1_stats import (PAIR_SEASON, REFERENCE, REG_CHANGE_SEASONS, TEAM_LINEAGE,
                      estimate_tau2, find_file, fit_ratings, iqr_mean, load_gaps,
                      pair_season_table, predicted_gap, session_gap_to_fastest, short)

Z95 = 1.96


def load_base() -> dict:
    """Gaps (with outlier flag), the cleaned qualifying table, pair-seasons and tau^2."""
    gaps = load_gaps()
    quali = pd.read_csv(find_file("qualifying_clean.csv"))
    ps = pair_season_table(gaps)
    tau2, _ = estimate_tau2(ps)
    usable = ps[ps["usable"]]
    # typical per-session spread of a pair's gap, for one-session ranges
    session_sd = float(usable["sd"].median())
    return {"gaps": gaps, "quali": quali, "ps": ps, "usable": usable, "tau2": tau2,
            "session_sd": session_sd, "seasons": sorted(gaps["year"].unique())}


def summary_numbers(base: dict) -> dict:
    gaps, ps = base["gaps"], base["ps"]
    return {"comparisons": len(gaps), "pair_seasons": len(ps),
            "usable": int(ps["usable"].sum()), "outliers": int(gaps["outlier"].sum()),
            "tau2": base["tau2"], "drivers": int(pd.concat([gaps["A"], gaps["B"]]).nunique()),
            "first": int(gaps["year"].min()), "last": int(gaps["year"].max()),
            "last_round": int(gaps.loc[gaps["year"] == gaps["year"].max(), "round"].max())}


def all_ratings(base: dict) -> tuple[pd.DataFrame, dict]:
    """Rating vs Russell for every driver who raced in each season (same as ratings_car_pace.ipynb)."""
    ps, gaps, tau2 = base["ps"], base["gaps"], base["tau2"]
    rows, fits = [], {}
    for year in base["seasons"]:
        fit = fit_ratings(ps, int(year), tau2)
        fits[int(year)] = fit
        raced = set(gaps.loc[gaps["year"] == year, "A"]) | set(gaps.loc[gaps["year"] == year, "B"])
        for d in sorted(raced & set(fit["theta"].index)):
            se = float(np.sqrt(fit["cov"].loc[d, d]))
            rows.append({"year": int(year), "driver": d, "name": short(d),
                         "rating": float(fit["theta"][d]), "se": se,
                         "ci_low": float(fit["theta"][d] - Z95 * se),
                         "ci_high": float(fit["theta"][d] + Z95 * se)})
    return pd.DataFrame(rows), fits


def driver_gap(fits: dict, year: int, a: str, b: str) -> tuple[float, float]:
    """Gap a minus b (% of lap time) for a season, and its SE, from the ratings fit."""
    return predicted_gap(fits[year], a, b)


def car_pace(base: dict, fits: dict) -> pd.DataFrame:
    """Car pace per team-season (Eq. 5), session-weighted over pairs (same as ratings_car_pace.ipynb)."""
    gaps, quali, usable = base["gaps"], base["quali"], base["usable"]
    gp = session_gap_to_fastest(gaps, quali)
    P = gp.groupby(PAIR_SEASON)["P_team"].apply(iqr_mean).unstack().reset_index()
    P = P.merge(usable[PAIR_SEASON + ["sessions"]], on=PAIR_SEASON)
    rows = []
    for r in P.itertuples():
        th, cov = fits[r.year]["theta"], fits[r.year]["cov"]
        if r.A in th.index and r.B in th.index:
            avg = (th[r.A] + th[r.B]) / 2
            var = (cov.loc[r.A, r.A] + cov.loc[r.B, r.B] + 2 * cov.loc[r.A, r.B]) / 4
            rows.append({"year": r.year, "team": r.team, "sessions": r.sessions,
                         "car_pace": r.mean - avg, "se": np.sqrt(r.se**2 + var)})
    pairs = pd.DataFrame(rows)

    def combine(d):
        w = d["sessions"]
        return pd.Series({"car_pace": np.average(d["car_pace"], weights=w),
                          "se": np.sqrt((w**2 * d["se"]**2).sum()) / w.sum()})

    car = pairs.groupby(["year", "team"]).apply(combine, include_groups=False).reset_index()
    car["ci_low"] = car["car_pace"] - Z95 * car["se"]
    car["ci_high"] = car["car_pace"] + Z95 * car["se"]
    car["lineage"] = car["team"].map(TEAM_LINEAGE).fillna(car["team"])
    car["team_name"] = car["lineage"].str.replace("-", " ").str.title()
    return car


def tracks(base: dict) -> pd.DataFrame:
    """Each circuit's most recent pole lap, to convert % gaps into seconds."""
    q = base["quali"].dropna(subset=["q3_seconds"])
    latest = q.groupby("circuit")["year"].max()
    q = q[q["year"] == q["circuit"].map(latest)]
    pole = q.groupby("circuit").agg(year=("year", "max"), pole_s=("q3_seconds", "min")).reset_index()
    pole["name"] = pole["circuit"].str.replace("-", " ").str.title()
    pole = pole[pole["year"] >= base["seasons"][-1] - 1]          # circuits on the recent calendar
    return pole.sort_values("name").reset_index(drop=True)


FLIP = ["prior_f", "last_f", "age_diff", "exp_diff", "rookie_diff", "newteam_diff"]
KEEP = ["has_prior", "has_last", "prior_se_f", "years_together", "reg_change"]
FEATURES = FLIP + KEEP


def _helpers(base: dict) -> dict:
    quali = base["quali"].copy()
    quali["lineage"] = quali["constructorId"].map(TEAM_LINEAGE).fillna(quali["constructorId"])
    return {
        "seasons_of": quali.groupby("driverId")["year"].apply(lambda s: sorted(set(s))),
        "age_at_start": quali.groupby(["driverId", "year"])["age"].min(),
        "team_of": quali.groupby(["driverId", "year"])["lineage"].agg(lambda s: s.mode().iat[0]),
        "together": base["gaps"].groupby(["A", "B"])["year"].apply(lambda s: sorted(set(s))),
    }


def feature_row(base: dict, h: dict, a: str, b: str, season: int, *, assume_same_team=False) -> dict:
    """Features for drivers a (A) and b (B) in `season`, using only earlier seasons.

    Works for seasons not yet raced (e.g. next season): age is last known age + the
    years since, and a team change is unknown, so it is taken as 0.
    """
    ps, usable, tau2 = base["ps"], base["usable"], base["tau2"]
    prior, prior_se = predicted_gap(fit_ratings(ps, season, tau2, use_future=False, reference=a), a, b)
    last = usable[(usable["year"] == season - 1) & (usable["A"] == a) & (usable["B"] == b)]["mean"]

    def age(d):
        if (d, season) in h["age_at_start"].index:
            return h["age_at_start"][(d, season)]
        last_year = max(y for (dd, y) in h["age_at_start"].index if dd == d)
        return h["age_at_start"][(d, last_year)] + (season - last_year)

    def exp(d):
        return sum(1 for y in h["seasons_of"][d] if y < season)

    def moved(d):
        if assume_same_team or (d, season) not in h["team_of"].index:
            return 0
        return int(h["team_of"].get((d, season - 1)) != h["team_of"].get((d, season)))

    row = {"A": a, "B": b, "year": season,
           "prior": prior, "prior_se": prior_se,
           "last": last.iat[0] if len(last) else np.nan,
           "age_diff": age(a) - age(b), "exp_diff": exp(a) - exp(b),
           "rookie_diff": int(exp(a) == 0) - int(exp(b) == 0),
           "newteam_diff": moved(a) - moved(b),
           "years_together": sum(1 for y in h["together"].get((a, b), []) if y < season),
           "reg_change": int(season in REG_CHANGE_SEASONS)}

    row["has_prior"] = int(not np.isnan(prior))
    row["has_last"] = int(not np.isnan(row["last"]))
    row["prior_f"] = 0.0 if np.isnan(prior) else prior
    row["prior_se_f"] = 1.0 if np.isnan(prior_se) else prior_se
    row["last_f"] = row["last"] if row["has_last"] else row["prior_f"]
    return row


def feature_table(base: dict) -> pd.DataFrame:
    """One row per usable pair-season from 2015 (same as ml_model.ipynb)."""
    h = _helpers(base)
    rows = []
    for r in base["usable"][base["usable"]["year"] >= base["seasons"][0] + 1].itertuples():
        row = feature_row(base, h, r.A, r.B, int(r.year))
        row.update({"team": r.team, "y": r.mean})
        rows.append(row)
    return pd.DataFrame(rows)


def _mirrored(d: pd.DataFrame) -> pd.DataFrame:
    m = d.copy()
    m[FLIP] = -m[FLIP]
    m["y"] = -m["y"]
    return pd.concat([d, m], ignore_index=True)


def _forest():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0)


def train_ml(base: dict) -> dict:
    """Hybrid model (last season's gap for returning pairs, Random Forest otherwise).

    Prediction interval = 95th percentile of the Hybrid's errors when each season from
    2016 on is predicted from the seasons before it (expanding window, as in ml_model.ipynb).
    """
    F = feature_table(base)
    first_val, last_season = base["seasons"][0] + 2, base["seasons"][-1]
    errors = []
    for s in range(first_val, last_season + 1):
        train, test = F[F["year"] < s], F[F["year"] == s]
        if test.empty or train.empty:
            continue
        rf = _forest().fit(_mirrored(train)[FEATURES], _mirrored(train)["y"])
        pred = np.where(test["has_last"] == 1, test["last"], rf.predict(test[FEATURES]))
        errors.extend(np.abs(test["y"] - pred))
    final = _forest().fit(_mirrored(F)[FEATURES], _mirrored(F)["y"])
    return {"features": F, "model": final, "half_width": float(np.quantile(errors, 0.95)),
            "n_val": len(errors), "helpers": _helpers(base)}


def predict_pair(base: dict, ml: dict, a: str, b: str, season: int) -> dict:
    """Hybrid prediction for drivers a and b as teammates in `season` (gap = a minus b)."""
    A, B = sorted([a, b])
    sign = 1 if a == A else -1
    row = feature_row(base, ml["helpers"], A, B, season, assume_same_team=True)
    X = pd.DataFrame([row])[FEATURES]
    rf = float(ml["model"].predict(X)[0])
    hybrid = row["last"] if row["has_last"] else rf
    return {"hybrid": sign * hybrid, "forest": sign * rf,
            "prior": None if not row["has_prior"] else sign * row["prior"],
            "last": None if not row["has_last"] else sign * row["last"],
            "used": "last season's gap" if row["has_last"] else "Random Forest",
            "half_width": ml["half_width"], "row": row}
