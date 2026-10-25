
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
PREDICTION_FILE = ROOT / "reports" / "latest_race_prediction.csv"

st.set_page_config(
    page_title="F1 Live Race Predictor",
    page_icon="🏎️",
    layout="wide",
)


@st.cache_data
def load_predictions(path, modified_time):
    return pd.read_csv(path)


st.title("F1 Race Predictor")
st.caption("Live race predictions powered by your trained Random Forest models.")

if not PREDICTION_FILE.exists():
    st.error(
        "No prediction file found. Run "
        "'python src\\predict_next_race.py' first."
    )
    st.stop()

# Include modification time in the cache key so refreshed files are reloaded.
modified_time = PREDICTION_FILE.stat().st_mtime

try:
    df = load_predictions(str(PREDICTION_FILE), modified_time)
except Exception as error:
    st.error(f"Could not load prediction file: {error}")
    st.stop()

required = [
    "race_name",
    "driver_name",
    "constructor_name",
    "qualifying_position",
    "predicted_finish_position",
    "podium_probability",
    "predicted_order",
]

missing = [column for column in required if column not in df.columns]

if missing:
    st.error(f"Prediction file is missing columns: {missing}")
    st.stop()

for column in [
    "qualifying_position",
    "predicted_finish_position",
    "podium_probability",
    "predicted_order",
]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(
    subset=["predicted_finish_position", "podium_probability"]
).copy()

if df.empty:
    st.warning("No usable predictions are available.")
    st.stop()

race_name = str(df["race_name"].iloc[0])
race_date = str(df["race_date"].iloc[0]) if "race_date" in df.columns else "Unknown"

st.subheader(race_name)
st.write(f"Scheduled date: {race_date}")

st.info(
    "Forecast generated from qualifying and historical race data. "
    "Grid penalties, weather, incidents, and race strategy may change "
    "the actual result. These probabilities are model estimates, not "
    "guaranteed or necessarily calibrated real-world probabilities."
)

# Summary metrics
col1, col2, col3 = st.columns(3)

col1.metric("Drivers predicted", len(df))
col2.metric(
    "Highest podium probability",
    f"{df['podium_probability'].max():.1%}",
)
col3.metric(
    "Average qualifying position",
    f"{df['qualifying_position'].mean():.1f}"
    if df["qualifying_position"].notna().any()
    else "Unavailable",
)

# Predicted finishing order
st.markdown("### Predicted finishing order")

order_df = df.sort_values("predicted_order").copy()

table_columns = [
    "predicted_order",
    "driver_name",
    "constructor_name",
    "qualifying_position",
    "predicted_finish_position",
    "podium_probability",
]

table = order_df[table_columns].copy()
table.columns = [
    "Predicted order",
    "Driver",
    "Constructor",
    "Qualifying position",
    "Estimated finishing position",
    "Podium probability",
]

table["Estimated finishing position"] = (
    table["Estimated finishing position"].round(1)
)
table["Podium probability"] = table["Podium probability"].map(
    lambda value: f"{value:.1%}"
)

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
)

# Podium probability chart
st.markdown("### Podium probabilities")

probability_df = df.sort_values("podium_probability", ascending=True)

fig = px.bar(
    probability_df,
    x="podium_probability",
    y="driver_name",
    color="constructor_name",
    orientation="h",
    labels={
        "podium_probability": "Estimated probability",
        "driver_name": "Driver",
        "constructor_name": "Constructor",
    },
    hover_data=["qualifying_position", "predicted_order"],
)

fig.update_layout(
    xaxis_tickformat=".0%",
    xaxis_title="Estimated podium probability",
    yaxis_title="",
    legend_title="Constructor",
)

st.plotly_chart(fig, use_container_width=True)

# Qualifying versus predicted position
st.markdown("### Qualifying versus predicted finish")

scatter_df = df.dropna(subset=["qualifying_position"]).copy()

if not scatter_df.empty:
    scatter = px.scatter(
        scatter_df,
        x="qualifying_position",
        y="predicted_finish_position",
        color="constructor_name",
        hover_name="driver_name",
        labels={
            "qualifying_position": "Qualifying position",
            "predicted_finish_position": "Predicted finishing position",
            "constructor_name": "Constructor",
        },
    )

    scatter.update_layout(
        xaxis=dict(autorange="reversed"),
        yaxis=dict(autorange="reversed"),
    )

    st.plotly_chart(scatter, use_container_width=True)
else:
    st.warning("Qualifying positions are not available for this prediction.")

# Download results
st.markdown("### Export predictions")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download predictions as CSV",
    data=csv_data,
    file_name="f1_race_predictions.csv",
    mime="text/csv",
)

st.caption(
    "F1 Race Predictor | Python · Pandas · Scikit-learn · Streamlit · Plotly"
)
