"""Shared functions for the CSD355 F1 teammate-gap project.

Used by 03_pair_season_stats.ipynb, 04_ratings_car_pace.ipynb and 05_ml_model.ipynb.
Every function here follows the method in the project report:
  Eq. 1  gap = 100 * (tA - tB) / tB                 (already in teammate_gaps.csv)
  Eq. 2  1.5 x IQR outlier rule, within each pair-season
  Eq. 3  t-interval for each pair-season mean
  Eq. 4  chains: gaps add, variances add, plus m * tau^2 for season drift
  Eq. 5  car pace = pair's gap to the fastest lap minus the drivers' ratings
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

PAIR_SEASON = ["year", "team", "A", "B"]
MIN_SESSIONS = 8          # report Table 4
REFERENCE = "george-russell"
REG_CHANGE_SEASONS = {2014, 2017, 2022, 2026}

# Teams that were renamed: map every name to the team's latest name, so a
# driver's "years at team" survives a rebrand.
TEAM_LINEAGE = {
    "toro-rosso": "racing-bulls", "alphatauri": "racing-bulls", "rb": "racing-bulls",
    "force-india": "aston-martin", "racing-point": "aston-martin",
    "sauber": "audi", "alfa-romeo": "audi", "kick-sauber": "audi",
    "lotus-f1": "alpine", "renault": "alpine",
    "marussia": "manor",
}


# ------------------------------------------------------------------- plot styling
BLUE, ORANGE = "#2a78d6", "#eb6834"          # series colours (blue first, orange second)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def setup_plot_style() -> None:
    """Consistent, quiet matplotlib style for every figure in the report."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "legend.frameon": False,
    })


def short(driver_id: str) -> str:
    """'george-russell' -> 'Russell', 'carlos-sainz-jr' -> 'Sainz'."""
    special = {"nyck-de-vries": "de Vries", "carlos-sainz-jr": "Sainz"}
    return special.get(driver_id, driver_id.split("-")[-1].title())


# --------------------------------------------------------------------------- files
def find_file(name: str) -> str:
    """Return the path of a data file, looking in data/ first, then the current folder."""
    for folder in ["data", ".", "..", os.path.join("..", "data")]:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} not found in data/ or the current folder")


def load_gaps() -> pd.DataFrame:
    """teammate_gaps.csv (one row per pair per race) with the IQR outlier flag added."""
    gaps = pd.read_csv(find_file("teammate_gaps.csv"))
    return add_outlier_flag(gaps)


# ------------------------------------------------------------------ Eq. 2: outliers
def add_outlier_flag(gaps: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Flag gaps outside [Q1 - k*IQR, Q3 + k*IQR] within each pair-season."""
    gaps = gaps.copy()
    grp = gaps.groupby(PAIR_SEASON)["gap"]
    q1 = grp.transform(lambda s: s.quantile(0.25))
    q3 = grp.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    gaps["outlier"] = (gaps["gap"] < q1 - k * iqr) | (gaps["gap"] > q3 + k * iqr)
    return gaps


# ------------------------------------------------------ Eq. 3: pair-season summary
def summarise_pair_season(s: pd.DataFrame, level: float = 0.95) -> pd.Series:
    """All statistics for one pair-season. `s` holds that pair-season's rows."""
    all_gaps = s["gap"]
    clean = s.loc[~s["outlier"], "gap"]
    n_all, n = len(all_gaps), len(clean)
    alpha = 1 - level

    mean, sd = clean.mean(), clean.std(ddof=1)
    se = sd / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    margin = t_crit * se

    # Head-to-head over all sessions (outliers included: a session is still won or lost)
    a_wins = int((all_gaps < 0).sum())
    p_hat = a_wins / n_all
    prop_ok = n_all * p_hat * (1 - p_hat) > 5          # normal approximation condition
    z = stats.norm.ppf(1 - alpha / 2)
    p_margin = z * np.sqrt(p_hat * (1 - p_hat) / n_all) if prop_ok else np.nan

    # Chi-square interval for the SD: (n-1)s^2 / sigma^2 ~ chi2(n-1)
    sd_low = np.sqrt((n - 1) * sd**2 / stats.chi2.ppf(1 - alpha / 2, n - 1))
    sd_high = np.sqrt((n - 1) * sd**2 / stats.chi2.ppf(alpha / 2, n - 1))

    return pd.Series({
        "sessions": n_all, "outliers": n_all - n, "n": n,
        "mean": mean, "median": all_gaps.median(), "sd": sd, "se": se,
        "ci_low": mean - margin, "ci_high": mean + margin, "margin": margin,
        "a_wins": a_wins, "b_wins": n_all - a_wins,
        "p_a": p_hat, "p_low": p_hat - p_margin, "p_high": p_hat + p_margin,
        "sd_low": sd_low, "sd_high": sd_high,
    })


def pair_season_table(gaps: pd.DataFrame) -> pd.DataFrame:
    """One row per pair-season; `usable` marks those with at least MIN_SESSIONS sessions."""
    ps = (gaps.groupby(PAIR_SEASON)
              .apply(summarise_pair_season, include_groups=False)
              .reset_index())
    ps["usable"] = ps["sessions"] >= MIN_SESSIONS
    ps["significant"] = (ps["ci_low"] > 0) | (ps["ci_high"] < 0)
    for c in ["sessions", "outliers", "n", "a_wins", "b_wins"]:
        ps[c] = ps[c].astype(int)
    return ps


def pair_gap(ps: pd.DataFrame, year: int, a: str, b: str) -> dict:
    """One pair-season written as 'a minus b' (flips the sign if a is not driver A)."""
    A, B = sorted([a, b])
    r = ps[(ps["year"] == year) & (ps["A"] == A) & (ps["B"] == B)].iloc[0]
    sign = 1 if a == A else -1
    lo, hi = sorted([sign * r["ci_low"], sign * r["ci_high"]])
    wins_a = int(r["a_wins"] if sign == 1 else r["b_wins"])
    return {"mean": sign * r["mean"], "se": r["se"], "ci_low": lo, "ci_high": hi,
            "wins_a": wins_a, "wins_b": int(r["sessions"]) - wins_a, "sessions": int(r["sessions"])}


# ------------------------------------------------------------- tau^2: season drift
def estimate_tau2(ps: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Season-to-season drift of a pair's gap, from pairs together in consecutive seasons.

    For a pair seen in seasons s and s+1:  d = g(s+1) - g(s)
    Var(d) = SE(s)^2 + SE(s+1)^2 + tau^2   ->   tau^2 = mean(d^2 - SE(s)^2 - SE(s+1)^2)
    tau^2 is per pair per season, which is exactly the term m * tau^2 in Eq. 4.
    """
    u = ps[ps["usable"]][["year", "A", "B", "mean", "se"]]
    nxt = u.assign(year=u["year"] - 1)
    both = u.merge(nxt, on=["year", "A", "B"], suffixes=("_s", "_next"))
    both["d"] = both["mean_next"] - both["mean_s"]
    both["excess"] = both["d"] ** 2 - both["se_s"] ** 2 - both["se_next"] ** 2
    return max(both["excess"].mean(), 0.0), both


# ------------------------------------------------------------ Eq. 4: simple chains
def chain(links: list[tuple[float, float]], seasons_crossed: int, tau2: float):
    """Chain gaps G_AC = G_AB + G_BC + ...; SE^2 adds, plus m * tau^2.

    links: list of (gap, se) in the direction of the chain.
    Returns (gap, se_without_drift, se_with_drift).
    """
    g = sum(l[0] for l in links)
    var = sum(l[1] ** 2 for l in links)
    return g, np.sqrt(var), np.sqrt(var + seasons_crossed * tau2)


# ------------------------------------------- driver ratings: weighted least squares
def fit_ratings(ps: pd.DataFrame, target: int, tau2: float, *,
                use_future: bool = True, max_distance: int | None = None,
                drop_pair: tuple[str, str] | None = None,
                reference: str = REFERENCE) -> dict:
    """Rate every driver relative to `reference` (= 0) for season `target`.

    Model:  mean gap of pair-season k  =  theta_A - theta_B + error
    Error variance = SE_k^2 + tau^2 * |target - year_k|   (older seasons count less,
    exactly the m * tau^2 idea of Eq. 4). Solved by weighted least squares, which
    uses every chain between two drivers at once instead of a single path.

    use_future=False uses only seasons before `target` (needed for the ML features,
    so the model never sees the season it predicts).
    Returns {'theta': Series, 'cov': DataFrame, 'scale': float}; drivers not linked
    to the reference get no rating.
    """
    import networkx as nx

    obs = ps[ps["usable"]].copy()
    obs = obs[obs["year"] <= target] if use_future else obs[obs["year"] < target]
    if max_distance is not None:
        obs = obs[(target - obs["year"]).abs() <= max_distance]
    if drop_pair is not None:
        a, b = sorted(drop_pair)
        obs = obs[~((obs["A"] == a) & (obs["B"] == b))]
    if use_future:   # descriptive ratings may also use later seasons
        later = ps[ps["usable"] & (ps["year"] > target)]
        if max_distance is not None:
            later = later[(later["year"] - target) <= max_distance]
        if drop_pair is not None:
            a, b = sorted(drop_pair)
            later = later[~((later["A"] == a) & (later["B"] == b))]
        obs = pd.concat([obs, later])

    graph = nx.Graph()
    graph.add_edges_from(zip(obs["A"], obs["B"]))
    if reference not in graph:
        return {"theta": pd.Series(dtype=float), "cov": pd.DataFrame(), "scale": np.nan}
    linked = nx.node_connected_component(graph, reference)
    obs = obs[obs["A"].isin(linked) & obs["B"].isin(linked)]

    drivers = sorted(linked - {reference})
    col = {d: i for i, d in enumerate(drivers)}
    X = np.zeros((len(obs), len(drivers)))
    for r, (a, b) in enumerate(zip(obs["A"], obs["B"])):
        if a in col:
            X[r, col[a]] = 1.0
        if b in col:
            X[r, col[b]] = -1.0
    y = obs["mean"].to_numpy()
    var = obs["se"].to_numpy() ** 2 + tau2 * (obs["year"] - target).abs().to_numpy()
    w = 1.0 / var

    xtwx = X.T @ (X * w[:, None])
    cov = np.linalg.pinv(xtwx)
    theta = cov @ (X.T @ (w * y))

    # If the data are noisier than the SEs say, widen the intervals (never narrow them)
    dof = len(y) - len(drivers)
    resid = y - X @ theta
    scale = max(1.0, float((w * resid**2).sum() / dof)) if dof > 0 else 1.0
    cov = cov * scale

    names = drivers + [reference]
    theta_s = pd.Series(np.append(theta, 0.0), index=names)
    cov_full = np.zeros((len(names), len(names)))
    cov_full[:-1, :-1] = cov
    return {"theta": theta_s, "cov": pd.DataFrame(cov_full, index=names, columns=names),
            "scale": scale}


def predicted_gap(fit: dict, a: str, b: str) -> tuple[float, float]:
    """Gap theta_a - theta_b and its SE from a fit_ratings result (NaN if not linked)."""
    th, cov = fit["theta"], fit["cov"]
    if a not in th.index or b not in th.index:
        return np.nan, np.nan
    var = cov.loc[a, a] + cov.loc[b, b] - 2 * cov.loc[a, b]
    return th[a] - th[b], float(np.sqrt(max(var, 0.0)))


# ------------------------------------------------------- Eq. 5: gap to fastest lap
def session_gap_to_fastest(gaps: pd.DataFrame, quali: pd.DataFrame) -> pd.DataFrame:
    """Add P_A, P_B: each driver's % gap to the fastest lap anyone set in the same segment.

    Comparing with the same segment (not always the pole lap) keeps both laps from the
    same session and track conditions; Q1 laps are slower than Q3 laps.
    """
    best = {seg: quali.groupby(["year", "round"])[f"{seg.lower()}_seconds"].min()
            for seg in ["Q1", "Q2", "Q3"]}
    ref = [best[seg][(y, r)] for y, r, seg in zip(gaps["year"], gaps["round"], gaps["segment"])]
    gaps = gaps.copy()
    gaps["ref"] = ref
    gaps["P_A"] = 100 * (gaps["tA"] - gaps["ref"]) / gaps["ref"]
    gaps["P_B"] = 100 * (gaps["tB"] - gaps["ref"]) / gaps["ref"]
    gaps["P_team"] = (gaps["P_A"] + gaps["P_B"]) / 2
    return gaps


def iqr_mean(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Mean and SE after removing values outside the 1.5 x IQR fences."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    kept = s[(s >= q1 - k * iqr) & (s <= q3 + k * iqr)]
    return pd.Series({"mean": kept.mean(), "se": kept.std(ddof=1) / np.sqrt(len(kept)),
                      "n": len(kept)})
