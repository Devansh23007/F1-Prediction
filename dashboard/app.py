
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = ROOT / "reports" / "2025_race_predictions.csv"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"


st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
)


@st.cache_data
def load_predictions():
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(PREDICTIONS_PATH)

    numeric_columns = [
        "predicted_order",
        "predicted_finish_position",
        "podium_probability",
        "finish_position",
        "podium",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}

    with open(METRICS_PATH, encoding="utf-8") as file:
        return json.load(file)


df = load_predictions()
metrics = load_metrics()

st.title("F1 Race Predictor")
st.caption(
    "Machine learning predictions for Formula 1 race finishing "
    "positions and podium probabilities."
)

if df.empty:
    st.error(
        "Prediction data not found. Run "
        "'python src\\train_models.py' from the project root first."
    )
    st.stop()

# Sidebar filters
st.sidebar.header("Race selection")

seasons = sorted(df["season"].dropna().unique().tolist())
selected_season = st.sidebar.selectbox(
    "Season",
    seasons,
    index=len(seasons) - 1,
)

season_df = df[df["season"] == selected_season].copy()

race_options = (
    season_df[["race_id", "race_name", "round"]]
    .drop_duplicates()
    .sort_values("round")
)

race_labels = {
    row["race_id"]: (
        f"Round {int(row['round'])}: {row['race_name']}"
    )
    for _, row in race_options.iterrows()
}

selected_race_id = st.sidebar.selectbox(
    "Grand Prix",
    race_options["race_id"].tolist(),
    format_func=lambda race_id: race_labels[race_id],
)

race_df = season_df[
    season_df["race_id"] == selected_race_id
].copy()

race_df = race_df.sort_values("predicted_order")

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Season", int(selected_season))
col2.metric("Drivers evaluated", len(race_df))
col3.metric(
    "Actual podiums in predicted top 3",
    int(
        race_df.nsmallest(3, "predicted_order")["podium"].sum()
    ),
)
col4.metric(
    "Mean predicted podium probability",
    f"{race_df['podium_probability'].mean():.1%}",
)

st.subheader(race_df["race_name"].iloc[0])

st.info(
    "These are historical test-set predictions using the available "
    "race data, including qualifying and grid information. "
    "They are not forecasts made before this race."
)

# Finishing order
st.markdown("### Predicted finishing order")

display_df = race_df[
    [
        "predicted_order",
        "driver_name",
        "constructor_name",
        "predicted_finish_position",
        "finish_position",
        "podium_probability",
    ]
].copy()

display_df.columns = [
    "Predicted order",
    "Driver",
    "Constructor",
    "Predicted position (regression)",
    "Actual position",
    "Podium probability",
]

display_df["Predicted position (regression)"] = (
    display_df["Predicted position (regression)"].round(1)
)

display_df["Podium probability"] = (
    display_df["Podium probability"].map(
        lambda value: f"{value:.1%}"
    )
)

st.dataframe(
    display_df,
    hide_index=True,
    use_container_width=True,
)

# Predicted versus actual order
st.markdown("### Predicted vs actual finishing position")

position_chart = px.scatter(
    race_df,
    x="finish_position",
    y="predicted_finish_position",
    hover_name="driver_name",
    color="constructor_name",
    labels={
        "finish_position": "Actual finishing position",
        "predicted_finish_position": "Predicted finishing position",
        "constructor_name": "Constructor",
    },
    title="Each point represents one driver",
)

position_chart.update_layout(
    xaxis=dict(autorange="reversed"),
    yaxis=dict(autorange="reversed"),
)

st.plotly_chart(position_chart, use_container_width=True)

# Podium probabilities
st.markdown("### Podium probability by driver")

probability_df = race_df.sort_values(
    "podium_probability", ascending=True
)

probability_chart = px.bar(
    probability_df,
    x="podium_probability",
    y="driver_name",
    color="constructor_name",
    orientation="h",
    hover_data=["finish_position", "predicted_order"],
    labels={
        "podium_probability": "Predicted probability",
        "driver_name": "Driver",
        "constructor_name": "Constructor",
    },
)

probability_chart.update_layout(
    xaxis_tickformat=".0%",
    yaxis_title="",
)

st.plotly_chart(probability_chart, use_container_width=True)

# Model evaluation
st.markdown("### Model evaluation")

if metrics:
    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Finishing position MAE",
        f"{metrics['finishing_position_mae']:.2f}",
    )
    m2.metric(
        "Podium ROC-AUC",
        (
            "N/A"
            if metrics.get("podium_roc_auc") is None
            else f"{metrics['podium_roc_auc']:.3f}"
        ),
    )
    m3.metric(
        "Podium Brier score",
        f"{metrics['podium_brier_score']:.3f}",
    )

    st.caption(
        f"Evaluation uses the held-out {metrics['test_season']} season. "
        "Metrics are calculated across the test dataset, not just "
        "the selected race."
    )

st.markdown("---")
st.caption(
    "F1 Race Predictor | Python · Pandas · Scikit-learn · Streamlit · Plotly"
)
