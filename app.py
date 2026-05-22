# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# =========================================================
# IMPORT CUSTOM MODULES
# =========================================================

from src.player_analysis import (
    get_top_batsmen,
    get_top_bowlers
)

from src.venue_analysis import (
    venue_win_stats
)

from src.toss_analysis import (
    toss_impact
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IPL 2026 Winner Predictor AI",
    page_icon="🏏",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: #FFD700;
}

.stButton>button {
    background-color: #FFD700;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_PATH = BASE_DIR / "data" / "processed"
MODELS_PATH = BASE_DIR / "models"
OUTPUTS_PATH = BASE_DIR / "outputs"


# =========================================================
# LOAD DATA
# =========================================================

advanced_dataset = pd.read_csv(
    PROCESSED_PATH /
    "advanced_match_dataset.csv"
)

team_features = pd.read_csv(
    PROCESSED_PATH /
    "team_features.csv"
)

prediction_results = pd.read_csv(
    OUTPUTS_PATH /
    "predictions" /
    "ipl_2026_predictions.csv"
)


# =========================================================
# LABEL ENCODERS
# =========================================================

encoders = {}

categorical_cols = [
    "team1",
    "team2",
    "toss_winner",
    "venue",
    "winner"
]

for col in categorical_cols:

    le = LabelEncoder()

    advanced_dataset[col] = le.fit_transform(
        advanced_dataset[col]
    )

    encoders[col] = le


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    MODELS_PATH / "xgboost.pkl"
)


# =========================================================
# TEAM LIST
# =========================================================

teams = sorted(
    team_features["team"].unique()
)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<h1 style='text-align: center; color: #FFD700;'>
🏏 IPL 2026 Winner Predictor AI
</h1>

<h4 style='text-align: center; color: white;'>
Powered by XGBoost + SHAP + Monte Carlo Simulation
</h4>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏆 Dashboard",
        "🔮 Match Predictor",
        "📊 IPL 2026 Winner Probability",
        "⚔ Team Comparison",
        "👤 Player Analytics",
        "🏟 Venue Analytics",
        "🪙 Toss Analytics",
        "📡 Live Match Center",
        "🧠 SHAP Analysis"
    ]
)

st.sidebar.info("""
AI-powered IPL analytics system.

Models Used:
- XGBoost
- LightGBM
- Random Forest

Features:
- Match Prediction
- Monte Carlo Simulation
- SHAP Explainability
- Player Analytics
- Venue Analytics
""")


# =========================================================
# TEAM FEATURES
# =========================================================

def get_team_features(team):

    row = team_features[
        team_features["team"] == team
    ]

    return row.iloc[0]


# =========================================================
# MATCH PREDICTION
# =========================================================

def predict_match(team1, team2, toss_winner):

    t1 = get_team_features(team1)
    t2 = get_team_features(team2)

    venue = random.choice(
        advanced_dataset["venue"].unique()
    )

    match_data = {

        "team1":
        encoders["team1"].transform([team1])[0],

        "team2":
        encoders["team2"].transform([team2])[0],

        "toss_winner":
        encoders["toss_winner"].transform(
            [toss_winner]
        )[0],

        "venue":
        int(venue),

        "team1_matches_played":
        t1["matches_played"],

        "team1_wins":
        t1["wins"],

        "team1_win_percentage":
        t1["win_percentage"],

        "team1_recent_form":
        t1["recent_form_wins"],

        "team1_total_runs":
        t1["total_runs"],

        "team1_balls_faced":
        t1["balls_faced"],

        "team1_strike_rate":
        t1["strike_rate"],

        "team1_wickets":
        t1["wickets"],

        "team1_balls_bowled":
        t1["balls_bowled"],

        "team1_economy":
        t1["economy"],

        "team2_matches_played":
        t2["matches_played"],

        "team2_wins":
        t2["wins"],

        "team2_win_percentage":
        t2["win_percentage"],

        "team2_recent_form":
        t2["recent_form_wins"],

        "team2_total_runs":
        t2["total_runs"],

        "team2_balls_faced":
        t2["balls_faced"],

        "team2_strike_rate":
        t2["strike_rate"],

        "team2_wickets":
        t2["wickets"],

        "team2_balls_bowled":
        t2["balls_bowled"],

        "team2_economy":
        t2["economy"],

        "win_percentage_diff":
        t1["win_percentage"] -
        t2["win_percentage"],

        "recent_form_diff":
        t1["recent_form_wins"] -
        t2["recent_form_wins"],

        "strike_rate_diff":
        t1["strike_rate"] -
        t2["strike_rate"],

        "economy_diff":
        t1["economy"] -
        t2["economy"],

        "wickets_diff":
        t1["wickets"] -
        t2["wickets"]
    }

    match_df = pd.DataFrame([match_data])

    probabilities = model.predict_proba(match_df)[0]

    prediction = np.argmax(probabilities)

    confidence = np.max(probabilities) * 100

    winner = encoders["winner"].inverse_transform(
        [prediction]
    )[0]

    return winner, confidence


# =========================================================
# DASHBOARD PAGE
# =========================================================

if page == "🏆 Dashboard":

    st.header("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Teams", len(teams))

    with col2:
        st.metric("Matches", "1146")

    with col3:
        st.metric("Simulations", "1000")

    with col4:
        st.metric("Best Accuracy", "54.3%")

    st.markdown("---")

    st.subheader("🏆 IPL 2026 Championship Probability")

    fig = px.pie(
        prediction_results,
        names="team",
        values="title_probability",
        title="IPL 2026 Winner Probability"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# MATCH PREDICTOR
# =========================================================

elif page == "🔮 Match Predictor":

    st.header("🔮 AI Match Winner Prediction")

    col1, col2 = st.columns(2)

    with col1:
        team1 = st.selectbox(
            "Select Team 1",
            teams
        )

    with col2:
        team2 = st.selectbox(
            "Select Team 2",
            teams
        )

    toss_winner = st.selectbox(
        "Select Toss Winner",
        [team1, team2]
    )

    if st.button("Predict Winner"):

        with st.spinner(
            "Running AI prediction..."
        ):

            time.sleep(2)

            winner, confidence = predict_match(
                team1,
                team2,
                toss_winner
            )

        st.success(
            f"🏆 {winner} has {confidence:.2f}% win probability"
        )

        loser_probability = 100 - confidence

        fig = go.Figure(data=[
            go.Bar(
                x=[winner, "Opponent"],
                y=[confidence, loser_probability]
            )
        ])

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.balloons()


# =========================================================
# IPL WINNER PROBABILITY
# =========================================================

elif page == "📊 IPL 2026 Winner Probability":

    st.header("📈 IPL 2026 Championship Probability")

    st.dataframe(
        prediction_results,
        use_container_width=True
    )

    fig = px.bar(
        prediction_results,
        x="team",
        y="title_probability",
        color="team",
        title="IPL 2026 Winner Probability"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "⬇ Download Prediction CSV",
        prediction_results.to_csv(index=False),
        "ipl_2026_predictions.csv",
        "text/csv"
    )


# =========================================================
# TEAM COMPARISON
# =========================================================

elif page == "⚔ Team Comparison":

    st.header("⚔ Team Comparison Analytics")

    col1, col2 = st.columns(2)

    with col1:
        team1 = st.selectbox(
            "Team 1",
            teams,
            key="compare1"
        )

    with col2:
        team2 = st.selectbox(
            "Team 2",
            teams,
            key="compare2"
        )

    t1 = get_team_features(team1)
    t2 = get_team_features(team2)

    comparison = pd.DataFrame({

        "Metric": [
            "Win %",
            "Strike Rate",
            "Wickets",
            "Economy"
        ],

        team1: [
            t1["win_percentage"],
            t1["strike_rate"],
            t1["wickets"],
            t1["economy"]
        ],

        team2: [
            t2["win_percentage"],
            t2["strike_rate"],
            t2["wickets"],
            t2["economy"]
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    radar_fig = go.Figure()

    radar_fig.add_trace(go.Scatterpolar(
        r=[
            t1["win_percentage"],
            t1["strike_rate"],
            t1["wickets"] / 10,
            10 - t1["economy"]
        ],
        theta=[
            "Win %",
            "Strike Rate",
            "Wickets",
            "Economy"
        ],
        fill='toself',
        name=team1
    ))

    radar_fig.add_trace(go.Scatterpolar(
        r=[
            t2["win_percentage"],
            t2["strike_rate"],
            t2["wickets"] / 10,
            10 - t2["economy"]
        ],
        theta=[
            "Win %",
            "Strike Rate",
            "Wickets",
            "Economy"
        ],
        fill='toself',
        name=team2
    ))

    st.plotly_chart(
        radar_fig,
        use_container_width=True
    )


# =========================================================
# PLAYER ANALYTICS
# =========================================================

elif page == "👤 Player Analytics":

    st.header("👤 Player Analytics")

    st.subheader("🏏 Top Batsmen")

    batsmen = get_top_batsmen()

    st.dataframe(
        batsmen,
        use_container_width=True
    )

    st.subheader("🎯 Top Bowlers")

    bowlers = get_top_bowlers()

    st.dataframe(
        bowlers,
        use_container_width=True
    )


# =========================================================
# VENUE ANALYTICS
# =========================================================

elif page == "🏟 Venue Analytics":

    st.header("🏟 Venue Analytics")

    venue_stats = venue_win_stats()

    st.dataframe(
        venue_stats,
        use_container_width=True
    )


# =========================================================
# TOSS ANALYTICS
# =========================================================

elif page == "🪙 Toss Analytics":

    st.header("🪙 Toss Impact Analysis")

    rate = toss_impact()

    st.metric(
        "Toss Winner Match Win Rate",
        f"{rate:.2f}%"
    )


# =========================================================
# LIVE MATCH CENTER
# =========================================================

elif page == "📡 Live Match Center":

    st.header("📡 Live IPL Match Center")

    st.info(
        "Live IPL API integration coming soon."
    )

    st.subheader("📅 IPL 2026 Schedule")

    st.markdown("""
    Upcoming IPL 2026 matches:
    - RCB vs SRH
    - MI vs KKR
    - RR vs CSK
    """)


# =========================================================
# SHAP ANALYSIS
# =========================================================

elif page == "🧠 SHAP Analysis":

    st.header("🧠 SHAP Explainability")

    shap_path = (
        OUTPUTS_PATH /
        "plots" /
        "shap_summary.png"
    )

    st.image(
        str(shap_path),
        caption="SHAP Feature Importance"
    )

    st.markdown("""
    ### 📌 Interpretation

    - Team identity strongly influences predictions.
    - Toss winner has moderate impact.
    - Venue contributes smaller effects.
    - Comparative team strength drives predictions.
    """)

    st.success(
        "SHAP Explainability integrated successfully!"
    )