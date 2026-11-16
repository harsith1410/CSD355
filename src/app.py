"""CSD355 F1 project dashboard.

Run from the project folder:
    pip install streamlit
    streamlit run src/app.py

Four tabs: driver ratings, teammate head-to-head, car pace, and a predictor.
Everything is recomputed from data/teammate_gaps.csv and qualifying_clean.csv with the
same functions as notebooks 03-05 (see dashboard_data.py and f1_stats.py).
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                      # so data/ and figures/ are found wherever you run it from
sys.path.insert(0, str(HERE))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import dashboard_data as dd
from f1_stats import BLUE, ORANGE, short

st.set_page_config(page_title="F1: driver vs car", layout="wide")
GREY = "#8a8984"
TEAM_COLOURS = ["#2a78d6", "#eb6834", "#1fa37a", "#9b5de5", "#d6a12a", "#d64f73", "#3fb5c9", "#6b7a2a"]


# ------------------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading the data and computing the ratings...")
def load_all():
    base = dd.load_base()
    ratings, fits = dd.all_ratings(base)
    car = dd.car_pace(base, fits)
    tracks = dd.tracks(base)
    return base, ratings, fits, car, tracks


@st.cache_resource(show_spinner="Training the ML model (about 10 seconds)...")
def load_ml():
    base = load_all()[0]
    return dd.train_ml(base)


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:                       # older Streamlit versions
        st.altair_chart(chart, use_container_width=True)


def show_table(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def faster_text(gap, a, b):
    """'Russell faster by 0.38%' for a gap written as a minus b."""
    if abs(gap) < 0.005:
        return "Level"
    return f"{short(a) if gap < 0 else short(b)} faster by {abs(gap):.2f}%"


def seconds(gap_pct, lap_s):
    return abs(gap_pct) / 100 * lap_s


base, ratings, fits, car, tracks = load_all()
numbers = dd.summary_numbers(base)
seasons = [int(s) for s in base["seasons"]]
latest = seasons[-1]

# ------------------------------------------------------------------------------- header
st.title("Separating driver skill from car performance in F1")
st.caption(f"Teammate qualifying gaps, {numbers['first']} to round {numbers['last_round']} of {numbers['last']}. "
           "Every number comes with a 95% interval. Data: F1DB (CC BY 4.0).")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Teammate comparisons", f"{numbers['comparisons']:,}")
c2.metric("Pair-seasons (≥ 8 sessions)", f"{numbers['usable']} of {numbers['pair_seasons']}")
c3.metric("Drivers", numbers["drivers"])
c4.metric("Outliers flagged", f"{numbers['outliers']}")
c5.metric("τ² (season drift)", f"{numbers['tau2']:.4f}")

tab_ratings, tab_h2h, tab_car, tab_predict = st.tabs(
    ["Driver ratings", "Teammate head-to-head", "Car pace", "Predictor"])

# ------------------------------------------------------------------------ driver ratings
with tab_ratings:
    st.subheader("How far is each driver from George Russell?")
    st.write("Rating = a driver's average qualifying gap to Russell, in % of lap time. "
             "Negative = faster than Russell. Lines show the 95% interval; if a line crosses 0, "
             "that driver can't be separated from Russell.")
    year = st.select_slider("Season", options=seasons, value=latest, key="rating_year")
    d = ratings[ratings["year"] == year].sort_values("rating")

    enc_y = alt.Y("name:N", sort=list(d["name"]), title=None)
    tips = [alt.Tooltip("name:N", title="Driver"),
            alt.Tooltip("rating:Q", title="Rating (%)", format="+.2f"),
            alt.Tooltip("ci_low:Q", title="95% CI low", format="+.2f"),
            alt.Tooltip("ci_high:Q", title="95% CI high", format="+.2f")]
    rules = alt.Chart(d).mark_rule(strokeWidth=2, color=BLUE).encode(
        y=enc_y, x=alt.X("ci_low:Q", title="% of lap time vs Russell (negative = faster)"),
        x2="ci_high:Q", tooltip=tips)
    dots = alt.Chart(d).mark_point(filled=True, size=90, color=BLUE, opacity=1).encode(
        y=enc_y, x="rating:Q", tooltip=tips)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GREY).encode(x="x:Q")
    show_chart((zero + rules + dots).properties(height=max(320, 26 * len(d))))

    st.markdown("**One driver over the years**")
    options = sorted(ratings["driver"].unique(), key=short)
    default = options.index("max-verstappen") if "max-verstappen" in options else 0
    who = st.selectbox("Driver", options, index=default, format_func=short, key="rating_driver")
    h = ratings[ratings["driver"] == who]
    band = alt.Chart(h).mark_area(opacity=0.18, color=BLUE).encode(
        x=alt.X("year:O", title="Season"), y=alt.Y("ci_low:Q", title="Rating vs Russell (%)"), y2="ci_high:Q")
    line = alt.Chart(h).mark_line(color=BLUE, strokeWidth=2).encode(x="year:O", y="rating:Q")
    pts = alt.Chart(h).mark_point(filled=True, size=70, color=BLUE, opacity=1).encode(
        x="year:O", y="rating:Q",
        tooltip=[alt.Tooltip("year:O", title="Season"), alt.Tooltip("rating:Q", format="+.2f", title="Rating (%)"),
                 alt.Tooltip("ci_low:Q", format="+.2f", title="CI low"),
                 alt.Tooltip("ci_high:Q", format="+.2f", title="CI high")])
    zero_y = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=GREY).encode(y="y:Q")
    show_chart((zero_y + band + line + pts).properties(height=300))
    st.caption("Ratings use all seasons, with older ones counting less (τ²), so they react slowly "
               "to a driver who improves quickly.")

    with st.expander("Table"):
        show_table(d[["name", "rating", "ci_low", "ci_high", "se"]].round(3).rename(
            columns={"name": "Driver", "rating": "Rating (%)", "ci_low": "95% CI low",
                     "ci_high": "95% CI high", "se": "SE"}))

# -------------------------------------------------------------------- teammate head-to-head
with tab_h2h:
    st.subheader("Teammate head-to-head")
    usable = base["usable"]
    c1, c2 = st.columns([1, 3])
    with c1:
        year = st.selectbox("Season", seasons[::-1], key="h2h_year")
    pairs = usable[usable["year"] == year].sort_values("team")
    labels = {i: f"{short(r.A)} vs {short(r.B)} ({r.team.replace('-', ' ').title()})"
              for i, r in pairs.iterrows()}
    with c2:
        pick = st.selectbox("Pair", list(labels), format_func=labels.get, key="h2h_pair")
    r = pairs.loc[pick]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean gap (outliers removed)", faster_text(r["mean"], r.A, r.B))
    m2.metric("95% interval", f"{r.ci_low:+.2f} to {r.ci_high:+.2f}%")
    wins_a, wins_b = int(r.a_wins), int(r.b_wins)
    m3.metric("Head-to-head", f"{short(r.A)} {wins_a}–{wins_b} {short(r.B)}")
    m4.metric("Sessions", f"{int(r.sessions)} ({int(r.outliers)} outliers)")
    verdict = ("The interval excludes 0: one driver was genuinely faster."
               if r.significant else "The interval includes 0: too close to call.")
    st.write(f"Gap written as {short(r.A)} minus {short(r.B)}: negative = {short(r.A)} faster. "
             f"Median of all sessions {r['median']:+.2f}%, session-to-session SD {r.sd:.2f}. {verdict}")

    s = base["gaps"][(base["gaps"]["year"] == year) & (base["gaps"]["A"] == r.A) & (base["gaps"]["B"] == r.B)].copy()
    circuits = base["quali"].drop_duplicates(["year", "round"]).set_index(["year", "round"])["circuit"]
    s["circuit"] = [circuits.get((y, rd), "") for y, rd in zip(s["year"], s["round"])]
    s["session"] = np.where(s["outlier"], "Outlier", "Clean")
    bars = alt.Chart(s).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("round:O", title="Round"),
        y=alt.Y("gap:Q", title=f"Gap, % ({short(r.A)} minus {short(r.B)})"),
        color=alt.Color("session:N", scale=alt.Scale(domain=["Clean", "Outlier"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("round:O", title="Round"), alt.Tooltip("circuit:N", title="Circuit"),
                 alt.Tooltip("segment:N", title="Segment"), alt.Tooltip("gap:Q", format="+.3f", title="Gap (%)"),
                 alt.Tooltip("session:N", title="Session")])
    mean_line = alt.Chart(pd.DataFrame({"y": [r["mean"]]})).mark_rule(color=GREY, strokeDash=[4, 4]).encode(y="y:Q")
    show_chart((bars + mean_line).properties(height=320))
    st.caption("Dashed line = the mean of the clean sessions. Outliers (1.5 × IQR rule) are shown but left out of the mean.")

# ------------------------------------------------------------------------------ car pace
with tab_car:
    st.subheader("Car pace with the driver effect removed")
    st.write("How far each car would be from the fastest lap with a Russell-level driver in both seats "
             "(% of lap time). Lower = faster car.")
    order = car.groupby("team_name")["car_pace"].mean().sort_values().index.tolist()
    heat = alt.Chart(car).mark_rect().encode(
        x=alt.X("year:O", title="Season"), y=alt.Y("team_name:N", sort=order, title=None),
        color=alt.Color("car_pace:Q", title="% slower",
                        scale=alt.Scale(scheme="blues", domain=[0, 2], clamp=True)),
        tooltip=[alt.Tooltip("team_name:N", title="Team"), alt.Tooltip("year:O", title="Season"),
                 alt.Tooltip("car_pace:Q", format=".2f", title="Car pace (%)"),
                 alt.Tooltip("ci_low:Q", format=".2f", title="95% CI low"),
                 alt.Tooltip("ci_high:Q", format=".2f", title="95% CI high")])
    labels_txt = alt.Chart(car).mark_text(fontSize=11).encode(
        x="year:O", y=alt.Y("team_name:N", sort=order),
        text=alt.Text("car_pace:Q", format=".2f"),
        color=alt.condition(alt.datum.car_pace > 1.1, alt.value("white"), alt.value("black")))
    show_chart((heat + labels_txt).properties(height=34 * len(order)))

    st.markdown("**Compare teams over time**")
    teams = st.multiselect("Teams", order, default=order[:3], key="car_teams")
    sel = car[car["team_name"].isin(teams)]
    if teams:
        colour = alt.Color("team_name:N", title=None,
                           scale=alt.Scale(domain=teams, range=TEAM_COLOURS[:len(teams)]))
        err = alt.Chart(sel).mark_rule(strokeWidth=1.5).encode(
            x=alt.X("year:O", title="Season"), y=alt.Y("ci_low:Q", title="Car pace (% slower than fastest lap)"),
            y2="ci_high:Q", color=colour)
        ln = alt.Chart(sel).mark_line(strokeWidth=2).encode(x="year:O", y="car_pace:Q", color=colour)
        pt = alt.Chart(sel).mark_point(filled=True, size=60, opacity=1).encode(
            x="year:O", y="car_pace:Q", color=colour,
            tooltip=[alt.Tooltip("team_name:N", title="Team"), alt.Tooltip("year:O", title="Season"),
                     alt.Tooltip("car_pace:Q", format=".2f", title="Car pace (%)")])
        show_chart((err + ln + pt).properties(height=320))

# ----------------------------------------------------------------------------- predictor
with tab_predict:
    st.subheader("Predictor")
    mode = st.radio("What do you want to predict?",
                    ["Any two drivers, on current form", f"Two drivers as teammates in {latest + 1} (ML model)"],
                    horizontal=True, key="mode")
    track_ids = tracks["circuit"].tolist()
    track_names = dict(zip(tracks["circuit"], tracks["name"]))
    pole = dict(zip(tracks["circuit"], tracks["pole_s"]))
    current = sorted(ratings.loc[ratings["year"] == latest, "driver"].unique(), key=short)

    c1, c2, c3 = st.columns(3)
    a = c1.selectbox("Driver 1", current, index=current.index("george-russell") if "george-russell" in current else 0,
                     format_func=short, key="p_a")
    b = c2.selectbox("Driver 2", current, index=current.index("max-verstappen") if "max-verstappen" in current else 1,
                     format_func=short, key="p_b")
    track = c3.selectbox("Track (for seconds)", track_ids,
                         index=track_ids.index("monza") if "monza" in track_ids else 0,
                         format_func=track_names.get, key="p_track")
    lap = pole[track]

    if a == b:
        st.warning("Pick two different drivers.")
    elif mode.startswith("Any"):
        g, se = dd.driver_gap(fits, latest, a, b)
        sd = base["session_sd"]
        avg_lo, avg_hi = g - 1.96 * se, g + 1.96 * se
        one = 1.96 * np.sqrt(se**2 + sd**2)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{latest} average gap", faster_text(g, a, b))
        m2.metric(f"In seconds at {track_names[track]}", f"{seconds(g, lap):.3f} s",
                  help=f"Pole lap used: {lap:.3f} s")
        m3.metric("95% interval (season average)", f"{avg_lo:+.2f} to {avg_hi:+.2f}%")
        st.write(f"For **one** qualifying session the gap could reasonably be anywhere from "
                 f"{g - one:+.2f}% to {g + one:+.2f}% ({short(a)} minus {short(b)}; negative = {short(a)} faster), "
                 f"because a single session swings about ±{sd:.2f}% around the average.")
        if avg_lo < 0 < avg_hi:
            st.info("The season-average interval includes 0: on this data the two can't be separated.")
        st.caption("Uses the weighted least-squares ratings (all teammate links at once, older seasons "
                   "counting less). Comparisons between drivers who never shared a car are rough: in the "
                   "backtest they named the faster driver about 7 times in 10.")
    else:
        ml = load_ml()
        p = dd.predict_pair(base, ml, a, b, latest + 1)
        hw = p["half_width"]
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Predicted {latest + 1} average gap", faster_text(p["hybrid"], a, b))
        m2.metric(f"In seconds at {track_names[track]}", f"{seconds(p['hybrid'], lap):.3f} s",
                  help=f"Pole lap used: {lap:.3f} s")
        m3.metric("95% prediction interval", f"{p['hybrid'] - hw:+.2f} to {p['hybrid'] + hw:+.2f}%")
        parts = [f"Method: **{p['used']}** (the Hybrid model chosen in notebook 05)."]
        if p["last"] is not None:
            parts.append(f"They were teammates in {latest}: gap {p['last']:+.2f}%.")
        parts.append(f"Random Forest on its own: {p['forest']:+.2f}%.")
        if p["prior"] is not None:
            parts.append(f"Ratings-only estimate: {p['prior']:+.2f}%.")
        st.write(" ".join(parts) + f" Gaps are {short(a)} minus {short(b)}; negative = {short(a)} faster.")
        st.caption(f"Assumes the two are teammates and neither changes team. The interval (±{hw:.2f}%) is the size "
                   f"of 95% of the model's past errors ({ml['n_val']} predictions, each made from earlier seasons only).")

st.divider()
st.caption("CSD355 Foundation of Data Science · Sanjith & Harsith N V · Methods in notebooks 03–05 "
           "(f1_stats.py, dashboard_data.py). One-lap qualifying pace only, not race pace.")
