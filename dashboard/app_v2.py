"""F1 Race Predictor - dashboard v2.

Run:  streamlit run dashboard/app_v2.py
Needs: python src/backtest_v2.py   (creates reports/ + data/processed/model_dataset_v2.csv)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_v2 as M  # noqa: E402
import live_predictions as LP  # noqa: E402

REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed" / "model_dataset_v2.csv"
RED, GREY, GREEN, BLUE = "#E10600", "#9AA0A6", "#2CB67D", "#4C8DFF"

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 10px 14px;
    border-radius: 10px; border-left: 4px solid #E10600;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data
def load_preds():
    p = REPORTS / "backtest_predictions.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_report(name):
    p = REPORTS / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_meta():
    p = REPORTS / "backtest_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def load_features():
    return pd.read_csv(DATA) if DATA.exists() else pd.DataFrame()


@st.cache_resource
def models_before(race_idx: int):
    df = load_features()
    return M.fit(df[df.race_idx < race_idx])


preds, meta = load_preds(), load_meta()
if preds.empty:
    st.error("No backtest data found. Run `python src/backtest_v2.py` first.")
    st.stop()

races = (preds.groupby(["race_id", "season", "round", "race_name"], as_index=False)
         .size().sort_values(["season", "round"]))


def race_picker(key, default_last=True, seasons=None):
    seasons = seasons or sorted(races.season.unique())
    c1, c2 = st.columns([1, 3])
    s = c1.selectbox("Season", seasons, index=len(seasons) - 1, key=f"s_{key}")
    sub = races[races.season == s]
    label = sub.apply(lambda r: f"R{r['round']:02d} - {r.race_name}", axis=1).tolist()
    pick = c2.selectbox("Race", label, index=len(label) - 1 if default_last else 0,
                        key=f"r_{key}")
    return sub.iloc[label.index(pick)].race_id


def podium_score(picks, actual):
    """3 pts exact position, 1 pt on podium in wrong position. Max 9."""
    return sum(3 if p == a else (1 if p in actual else 0)
               for p, a in zip(picks, actual))


def order_top3(g, col, asc=True):
    return g.sort_values(col, ascending=asc).driver_name.head(3).tolist()


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🏎️ F1 Race Predictor")
page = st.sidebar.radio("Page", ["⚡ Live Next Race", "🏁 Race Explorer", "🎯 Check Yourself",
                                 "📊 Model Report", "🔮 What-if Lab"])
st.sidebar.caption(
    "All predictions on these pages are **walk-forward**: each race was predicted "
    "by a model that had only seen earlier races.")


# ================================================================== LIVE NEXT RACE
if page == "⚡ Live Next Race":
    st.title("⚡ Live Race Predictor & Prediction Log")
    st.write(
        "Real-time prediction for the upcoming Grand Prix on the F1 calendar. "
        "Predictions are saved to a persistent **SQLite database** (`data/predictions_log.db`) "
        "with timestamps for transparency and post-race audit."
    )

    LP.init_db()
    LP.seed_recent_predictions_if_empty()

    upcoming = LP.get_upcoming_round(2026)
    s, r, rname = upcoming["season"], upcoming["round"], upcoming["race_name"]
    circuit = upcoming["circuit_id"]

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"Next Event: Round {r} - {rname}")
        st.caption(f"Circuit: **{circuit}** | Date: **{upcoming.get('date', 'TBD')}**")
    with col2:
        st.metric("Season", s)
    with col3:
        st.metric("Round", f"R{r:02d}")

    # Check for live qualifying
    qual_df = None
    if st.button("🔄 Check Live Qualifying on Jolpica / FastF1"):
        qual_df = LP.fetch_live_qualifying(s, r)
        if qual_df is not None and not qual_df.empty:
            st.success(f"Fetched official qualifying grid with {len(qual_df)} drivers!")
        else:
            st.info("Official qualifying not published yet. Using latest driver form grid.")

    if qual_df is None or qual_df.empty:
        grid_base = LP.get_default_grid_for_next_round(s)
    else:
        grid_base = qual_df

    st.write("### 1. Starting Grid for this Grand Prix")
    st.caption("You can edit starting positions or penalty drops before running predictions.")
    
    edited_grid = st.data_editor(
        grid_base[["grid_position", "driver_name", "constructor_name"]].rename(
            columns={"grid_position": "Grid", "driver_name": "Driver", "constructor_name": "Team"}
        ),
        disabled=["Driver", "Team"],
        hide_index=True,
        width="stretch",
        height=32 * len(grid_base) + 38,
        key=f"grid_live_{s}_{r}"
    )

    grid_input = grid_base.copy()
    grid_input["grid_position"] = edited_grid["Grid"].values

    c_run, c_temp = st.columns([1, 2])
    with c_run:
        run_btn = st.button("🚀 Run Live Prediction & Log to Database", type="primary")
    with c_temp:
        temp_scale = st.slider("Race Chaos Multiplier", 0.6, 1.8, 1.0, 0.1, key="live_chaos")

    if run_btn:
        with st.spinner("Training walk-forward model on historical data & running 12,000 Monte Carlo simulations..."):
            live_res = LP.run_prediction_for_grid(grid_input, s, r, rname, circuit, temp_scale=temp_scale)
            run_id = LP.log_predictions_to_db(live_res, notes=f"Dashboard live run (chaos={temp_scale})")
            st.session_state[f"latest_live_preds_{s}_{r}"] = live_res
            st.success(f"Predictions calculated & logged to SQLite database! (Run ID: #{run_id})")

    # Display prediction if in state
    if f"latest_live_preds_{s}_{r}" in st.session_state:
        live_res = st.session_state[f"latest_live_preds_{s}_{r}"]
        st.write("### 2. Predicted Finishing Order & Probabilities")
        
        # Top 3 cards
        top3 = live_res.nsmallest(3, "pred_position")
        c1, c2, c3 = st.columns(3)
        c1.metric("🥇 Predicted Winner", top3.iloc[0]["driver_name"], f"{top3.iloc[0]['p_win']*100:.1f}% Win Probability")
        c2.metric("🥈 P2 Prediction", top3.iloc[1]["driver_name"], f"{top3.iloc[1]['p_podium']*100:.1f}% Podium Probability")
        c3.metric("🥉 P3 Prediction", top3.iloc[2]["driver_name"], f"{top3.iloc[2]['p_podium']*100:.1f}% Podium Probability")

        l_tab, r_tab = st.columns([3, 2])
        with l_tab:
            show = live_res[["pred_position", "driver_name", "constructor_name", "grid_position", "p_win", "p_podium", "p_top10", "exp_position"]].copy()
            show["p_win"] = (show.p_win * 100).round(1)
            show["p_podium"] = (show.p_podium * 100).round(1)
            show["p_top10"] = (show.p_top10 * 100).round(1)
            show["exp_position"] = show.exp_position.round(2)
            show["gain_vs_grid"] = (show.grid_position - show.pred_position).astype(int)
            show.columns = ["Pred Pos", "Driver", "Team", "Grid", "Win %", "Podium %", "Top 10 %", "Exp Finish", "Gain vs Grid"]
            st.dataframe(show, hide_index=True, width="stretch", height=32 * len(show) + 38)
        
        with r_tab:
            top_prob = live_res.nlargest(10, "p_podium").sort_values("p_podium")
            fig = go.Figure(go.Bar(
                x=top_prob.p_podium * 100, y=top_prob.driver_name, orientation="h",
                marker_color=RED,
                text=(top_prob.p_podium * 100).round(1).astype(str) + "%",
                textposition="outside"
            ))
            fig.update_layout(title="Top 10 Podium Probabilities", xaxis_title="Probability (%)", height=420, margin=dict(l=0, r=20, t=50, b=0))
            st.plotly_chart(fig, width="stretch")

    # SQLite Database Log Viewer
    st.divider()
    st.write("### 3. Persistent SQLite Prediction Log (`data/predictions_log.db`)")
    log_df = LP.get_logged_predictions()
    if not log_df.empty:
        log_runs = log_df[["season", "round", "race_name", "timestamp"]].drop_duplicates()
        st.write(f"Logged Runs Count: **{len(log_runs)}** | Total Driver Predictions Logged: **{len(log_df)}**")
        
        # Formatted log display
        disp = log_df.copy()
        disp["p_win"] = (disp.p_win * 100).round(1).astype(str) + "%"
        disp["p_podium"] = (disp.p_podium * 100).round(1).astype(str) + "%"
        disp["actual_position"] = disp["actual_position"].fillna("Pending")
        disp.columns = ["ID", "Timestamp", "Season", "Round", "Race", "Driver", "Team", "Grid", "Pred", "Win %", "Podium %", "Actual", "Scored"]
        st.dataframe(disp.head(30), hide_index=True, width="stretch")
    else:
        st.info("No predictions logged in database yet.")

# ================================================================== EXPLORER
elif page == "🏁 Race Explorer":
    st.title("Race Explorer")
    rid = race_picker("explore")
    g = preds[preds.race_id == rid].sort_values("pred_order").copy()
    race_name, season = g.race_name.iloc[0], g.season.iloc[0]
    st.subheader(f"{season} {race_name}")

    act3 = set(g[g.finish_position <= 3].driver_name)
    mod3 = set(g.nsmallest(3, "pred_order").driver_name)
    grid3 = set(g.nsmallest(3, "grid_position").driver_name)
    winner = g.loc[g.finish_position == 1, "driver_name"].iloc[0]
    pred_winner = g.loc[g.pred_order == 1, "driver_name"].iloc[0]
    c = st.columns(4)
    c[0].metric("Actual winner", winner)
    c[1].metric("Model's pick", pred_winner, "✔ correct" if pred_winner == winner else "✘ wrong",
                delta_color="normal" if pred_winner == winner else "inverse")
    c[2].metric("Model podium hits", f"{len(mod3 & act3)} / 3")
    c[3].metric("Grid-order podium hits", f"{len(grid3 & act3)} / 3")

    left, right = st.columns([3, 2])
    with left:
        show = g[["pred_order", "driver_name", "constructor_name", "grid_position",
                  "finish_position", "p_win", "p_podium", "status"]].copy()
        show["p_win"] = (show.p_win * 100).round(1)
        show["p_podium"] = (show.p_podium * 100).round(1)
        show["error"] = (show.pred_order - show.finish_position).astype(int)
        show.columns = ["Pred", "Driver", "Team", "Grid", "Actual", "Win %", "Podium %",
                        "Status", "Pred-Actual"]
        st.dataframe(show, hide_index=True, width="stretch", height=36 * 21)
    with right:
        top = g.nlargest(10, "p_podium").sort_values("p_podium")
        fig = go.Figure(go.Bar(
            x=top.p_podium * 100, y=top.driver_name, orientation="h",
            marker_color=[GREEN if d in act3 else RED for d in top.driver_name],
            text=(top.p_podium * 100).round(0).astype(int).astype(str) + "%",
            textposition="outside"))
        fig.update_layout(title="Podium probability (green = actually on podium)",
                          xaxis_title="%", height=420, margin=dict(l=0, r=20, t=50, b=0))
        st.plotly_chart(fig, width="stretch")

    fig = px.scatter(g, x="pred_order", y="finish_position", hover_name="driver_name",
                     color="constructor_name", labels={"pred_order": "Predicted position",
                                                       "finish_position": "Actual position"})
    fig.add_trace(go.Scatter(x=[1, 20], y=[1, 20], mode="lines",
                             line=dict(color=GREY, dash="dash"), name="perfect"))
    fig.update_layout(height=380, title="Predicted vs actual finishing position")
    st.plotly_chart(fig, width="stretch")

# ================================================================== CHECK YOURSELF
elif page == "🎯 Check Yourself":
    st.title("🎯 Check Yourself")
    st.write("Pick a past race, predict the podium from the starting grid, then see how you "
             "score against the **model** and the **'just copy the grid'** baseline.")

    ss = st.session_state
    ss.setdefault("score", {"You": 0, "Model": 0, "Grid": 0, "rounds": 0})
    ss.setdefault("revealed", set())

    mode = st.radio("Race", ["Random race", "I'll choose"], horizontal=True)
    if mode == "Random race":
        if "rand_race" not in ss or st.button("🎲 New random race"):
            ss.rand_race = races.sample(1).race_id.iloc[0]
        rid = ss.rand_race
    else:
        rid = race_picker("game")

    g = preds[preds.race_id == rid].sort_values("grid_position").copy()
    rname = f"{g.season.iloc[0]} {g.race_name.iloc[0]}"
    st.subheader("🔒 Hidden race" if mode == "Random race" and rid not in ss.revealed else rname)
    if mode == "Random race" and rid not in ss.revealed:
        st.caption("Race name hidden so you can't just remember the result. Drivers & grid below are real.")

    view = g[["grid_position", "driver_name", "constructor_name", "driver_elo", "team_elo"]].copy()
    view["driver_elo"] = view.driver_elo.round(0).astype(int)
    view["team_elo"] = view.team_elo.round(0).astype(int)
    view.columns = ["Grid", "Driver", "Team", "Driver Elo", "Team Elo"]
    st.dataframe(view, hide_index=True, width="stretch", height=36 * 21)
    st.caption("Elo = rating built from previous races only (higher = stronger).")

    names = g.driver_name.tolist()
    c1, c2, c3 = st.columns(3)
    p1 = c1.selectbox("🥇 P1", names, index=0, key=f"p1_{rid}")
    p2 = c2.selectbox("🥈 P2", names, index=min(1, len(names) - 1), key=f"p2_{rid}")
    p3 = c3.selectbox("🥉 P3", names, index=min(2, len(names) - 1), key=f"p3_{rid}")

    if st.button("Lock in & reveal", type="primary"):
        if len({p1, p2, p3}) < 3:
            st.warning("Pick three different drivers.")
        elif rid in ss.revealed:
            st.info("You already played this race. Try another.")
        else:
            actual = g.sort_values("finish_position").driver_name.head(3).tolist()
            mine = [p1, p2, p3]
            model = order_top3(g, "pred_order")
            grid = order_top3(g, "grid_position")
            s = {"You": podium_score(mine, actual), "Model": podium_score(model, actual),
                 "Grid": podium_score(grid, actual)}
            for k, v in s.items():
                ss.score[k] += v
            ss.score["rounds"] += 1
            ss.revealed.add(rid)
            st.success(f"**{rname}**  -  actual podium: 🥇 {actual[0]}  🥈 {actual[1]}  🥉 {actual[2]}")
            cc = st.columns(3)
            cc[0].metric("You", f"{s['You']} / 9", ", ".join(mine))
            cc[1].metric("Model", f"{s['Model']} / 9", ", ".join(model))
            cc[2].metric("Grid copy", f"{s['Grid']} / 9", ", ".join(grid))
            st.caption("Scoring: 3 pts per driver in the exact podium spot, 1 pt for a podium driver in the wrong spot.")

    if ss.score["rounds"]:
        st.divider()
        n = ss.score["rounds"]
        st.subheader(f"Scoreboard after {n} race{'s' if n > 1 else ''}")
        sc = pd.DataFrame({"Player": ["You", "Model", "Grid copy"],
                           "Total": [ss.score["You"], ss.score["Model"], ss.score["Grid"]]})
        sc["Avg per race"] = (sc.Total / n).round(2)
        st.dataframe(sc, hide_index=True)
        st.plotly_chart(px.bar(sc, x="Player", y="Avg per race", color="Player",
                               color_discrete_sequence=[RED, BLUE, GREY], range_y=[0, 9]),
                        width="stretch")
        if st.button("Reset scoreboard"):
            ss.score = {"You": 0, "Model": 0, "Grid": 0, "rounds": 0}
            ss.revealed = set()
            st.rerun()

# ================================================================== MODEL REPORT
elif page == "📊 Model Report":
    st.title("Model Report")
    st.write("Is the model actually better than the obvious baseline? **Hold-out = 2024-25** "
             "(never used to choose anything). Every race is predicted by a model trained only on earlier races.")
    hold = load_report("backtest_summary_holdout.csv")
    allr = load_report("backtest_summary_all.csv")
    per_race = load_report("backtest_per_race.csv")
    calib = load_report("calibration_podium.csv")

    def tidy(d):
        d = d.copy()
        d.columns = ["Model", "Rank MAE ↓", "Spearman ↑", "Winner hit", "Top-3 hits (of 3) ↑",
                     "Exact podiums", "Top-10 hits (of 10) ↑", "Races"]
        return d.round(3)

    m = hold.set_index("model")
    mod, grd = m.loc["Grid + learned correction"], m.loc["Grid order (baseline)"]
    c = st.columns(4)
    c[0].metric("Spearman rank corr.", f"{mod.spearman:.3f}", f"{mod.spearman - grd.spearman:+.3f} vs grid")
    c[1].metric("Avg position error", f"{mod.rank_mae:.2f}", f"{mod.rank_mae - grd.rank_mae:+.2f} vs grid",
                delta_color="inverse")
    c[2].metric("Podium drivers found", f"{mod.top3_hits:.2f} / 3", f"{mod.top3_hits - grd.top3_hits:+.2f} vs grid")
    c[3].metric("Exact podiums", f"{int(mod.exact_podium)} / {int(mod.races)}",
                f"{int(mod.exact_podium - grd.exact_podium):+d} vs grid")

    t1, t2 = st.tabs(["Hold-out 2024-25", "All walk-forward 2022-25"])
    t1.dataframe(tidy(hold), hide_index=True, width="stretch")
    t2.dataframe(tidy(allr), hide_index=True, width="stretch")
    st.info("**Honest reading:** the starting grid is already a very strong predictor in modern F1. "
            "The model only beats it by a small but consistent margin - that is realistic. "
            "A model claiming ~97% accuracy usually means it predicts 'not the winner' for everyone.")

    st.subheader("Skill by season (Spearman correlation, higher is better)")
    ps = per_race.groupby(["season", "model"], as_index=False).spearman.mean()
    fig = px.bar(ps, x="season", y="spearman", color="model", barmode="group",
                 color_discrete_map={"Grid order (baseline)": GREY, "Combined Elo (baseline)": BLUE,
                                     "Grid + learned correction": RED})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Cumulative podium hits across races")
    cum = per_race.sort_values("race_id").copy()
    cum["cum_hits"] = cum.groupby("model").top3_hits.cumsum()
    cum["race_no"] = cum.groupby("model").cumcount() + 1
    st.plotly_chart(px.line(cum, x="race_no", y="cum_hits", color="model",
                            color_discrete_map={"Grid order (baseline)": GREY,
                                                "Combined Elo (baseline)": BLUE,
                                                "Grid + learned correction": RED}),
                    width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Are the probabilities honest?")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                                 line=dict(color=GREY, dash="dash")))
        fig.add_trace(go.Scatter(x=calib.predicted, y=calib.actual, mode="lines+markers",
                                 name="model", marker=dict(size=calib.n.clip(upper=60) / 4 + 6,
                                                           color=RED)))
        fig.update_layout(xaxis_title="Predicted podium probability",
                          yaxis_title="Actual podium rate", height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("Closer to the dashed line = better calibrated. Bubble size = number of driver-races.")
    with right:
        st.subheader("Probability quality (Brier, lower is better)")
        pm = meta.get("prob_metrics", {})
        if pm:
            tbl = pd.DataFrame({
                "Target": ["Podium", "Win"],
                "Constant rate": [pm["podium_brier_constant"], pm["win_brier_constant"]],
                "Grid baseline": [pm["podium_brier_grid_baseline"], pm["win_brier_grid_baseline"]],
                "Model": [pm["podium_brier_model"], pm["win_brier_model"]]}).round(4)
            st.dataframe(tbl, hide_index=True)
            st.caption("The model's probabilities are on par with a grid-based baseline - "
                       "both far better than guessing a constant rate.")

# ================================================================== WHAT-IF
else:
    st.title("🔮 What-if Lab")
    st.write("Change the starting grid (penalty, pit-lane start, a driver qualifying P1...) "
             "and see how win / podium chances move. The model is re-fitted using only races before the one you pick.")
    feats = load_features()
    rid = race_picker("whatif")
    race_idx = int(feats.loc[feats.race_id == rid, "race_idx"].iloc[0])
    base = feats[feats.race_id == rid].sort_values("grid_position").reset_index(drop=True)

    left, right = st.columns([2, 3])
    with left:
        st.caption("Edit the **Grid** column (1 = pole). Duplicates are allowed - ties are broken by row order.")
        edit = st.data_editor(
            base[["driver_name", "constructor_name", "grid_position"]].rename(
                columns={"driver_name": "Driver", "constructor_name": "Team", "grid_position": "Grid"}),
            disabled=["Driver", "Team"], hide_index=True, width="stretch", height=36 * 21,
            key=f"ed_{rid}")
        temp_scale = st.slider("Race chaos (higher = more random)", 0.5, 2.0, 1.0, 0.1)
        wet = st.checkbox("Extra reliability risk (+50% DNF probability)")

    scen = base.copy()
    scen["grid_position"] = edit["Grid"].astype(float).values
    n = len(scen)
    scen["grid_penalty"] = scen.grid_position - scen.qualifying_position
    scen["grid_rank_pct"] = scen.grid_position / n
    scen["pit_lane_start"] = (scen.grid_position >= n + 1).astype(int)

    models = models_before(race_idx)
    sc = M.score(models, scen)
    pdnf = M.dnf_prob(scen, models["base_dnf"]) * (1.5 if wet else 1.0)
    base_sc = M.score(models, base)
    T = meta.get("temperature_model", 1.2) * temp_scale
    P = M.simulate(sc, np.clip(pdnf, 0, 0.8), T, n=12000)
    P0 = M.simulate(base_sc, M.dnf_prob(base, models["base_dnf"]), meta.get("temperature_model", 1.2), n=12000)
    S, S0 = M.summarise_probs(P), M.summarise_probs(P0)

    res = pd.DataFrame({"Driver": base.driver_name, "Team": base.constructor_name,
                        "Win %": S["p_win"] * 100, "Win % (original)": S0["p_win"] * 100,
                        "Podium %": S["p_podium"] * 100, "Podium % (original)": S0["p_podium"] * 100,
                        "Exp. finish": S["exp_position"]})
    res["Δ Podium"] = res["Podium %"] - res["Podium % (original)"]
    res = res.sort_values("Podium %", ascending=False)
    with right:
        top = res.head(10).melt(id_vars="Driver", value_vars=["Podium % (original)", "Podium %"],
                                var_name="Scenario", value_name="Podium probability %")
        top["Scenario"] = top.Scenario.map({"Podium % (original)": "Original grid", "Podium %": "Your scenario"})
        st.plotly_chart(px.bar(top, y="Driver", x="Podium probability %", color="Scenario",
                               orientation="h", barmode="group",
                               color_discrete_map={"Original grid": GREY, "Your scenario": RED},
                               category_orders={"Driver": res.head(10).Driver.tolist()[::-1]}
                               ).update_layout(height=480), width="stretch")
    st.dataframe(res.round(1), hide_index=True, width="stretch")
    actual = base.sort_values("finish_position").driver_name.head(3).tolist()
    with st.expander("Peek at what really happened"):
        st.write(" / ".join(f"{i + 1}. {d}" for i, d in enumerate(actual)))
