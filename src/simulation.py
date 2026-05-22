# src/simulation.py

import pandas as pd
import numpy as np
import joblib
import random

from pathlib import Path
from sklearn.preprocessing import LabelEncoder


# -----------------------------------
# PROJECT PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_PATH = BASE_DIR / "data" / "processed"
MODELS_PATH = BASE_DIR / "models"
OUTPUTS_PATH = BASE_DIR / "outputs" / "predictions"


# -----------------------------------
# LOAD DATASET
# -----------------------------------

dataset = pd.read_csv(
    PROCESSED_PATH /
    "advanced_match_dataset.csv"
)


# -----------------------------------
# ENCODE CATEGORICAL FEATURES
# -----------------------------------

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

    dataset[col] = le.fit_transform(
        dataset[col]
    )

    encoders[col] = le


# -----------------------------------
# LOAD TRAINED MODEL
# -----------------------------------

model = joblib.load(
    MODELS_PATH / "xgboost.pkl"
)


# -----------------------------------
# IPL TEAMS
# -----------------------------------

teams = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Lucknow Super Giants",
    "Gujarat Titans"
]


# -----------------------------------
# LOAD TEAM FEATURES
# -----------------------------------

team_stats = pd.read_csv(
    PROCESSED_PATH /
    "team_features.csv"
)


# -----------------------------------
# GET TEAM FEATURES
# -----------------------------------

def get_team_features(team):

    row = team_stats[
        team_stats["team"] == team
    ]

    if row.empty:
        return None

    return row.iloc[0]


# -----------------------------------
# SIMULATE SINGLE MATCH
# -----------------------------------

def simulate_match(team1, team2):

    toss_winner = random.choice([team1, team2])

    venue = random.choice(
        dataset["venue"].unique()
    )

    t1 = get_team_features(team1)
    t2 = get_team_features(team2)

    if t1 is None or t2 is None:
        return random.choice([team1, team2])

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

    prediction = model.predict(match_df)[0]

    winner = encoders["winner"].inverse_transform(
        [prediction]
    )[0]

    return winner


# -----------------------------------
# SIMULATE IPL SEASON
# -----------------------------------

def simulate_season():

    points_table = {
        team: 0 for team in teams
    }

    for i in range(len(teams)):

        for j in range(i + 1, len(teams)):

            team1 = teams[i]
            team2 = teams[j]

            # Home and away matches
            for _ in range(2):

                winner = simulate_match(
                    team1,
                    team2
                )

                points_table[winner] += 2

    return points_table


# -----------------------------------
# MONTE CARLO SIMULATION
# -----------------------------------

def monte_carlo_simulation(n=1000):

    print(f"\nRunning {n} simulations...\n")

    title_wins = {
        team: 0 for team in teams
    }

    for sim in range(n):

        table = simulate_season()

        champion = max(
            table,
            key=table.get
        )

        title_wins[champion] += 1

        if sim % 100 == 0:
            print(f"Completed {sim} simulations")

    probabilities = pd.DataFrame({

        "team": title_wins.keys(),

        "titles_won": title_wins.values()
    })

    probabilities["title_probability"] = (
        probabilities["titles_won"] / n
    ) * 100

    probabilities = probabilities.sort_values(
        "title_probability",
        ascending=False
    )

    return probabilities


# -----------------------------------
# MAIN
# -----------------------------------

if __name__ == "__main__":

    results = monte_carlo_simulation(1000)

    print("\nIPL 2026 Title Probabilities:\n")

    print(results)

    save_path = (
        OUTPUTS_PATH /
        "ipl_2026_predictions.csv"
    )

    results.to_csv(
        save_path,
        index=False
    )

    print(f"\nPredictions saved at:\n{save_path}")