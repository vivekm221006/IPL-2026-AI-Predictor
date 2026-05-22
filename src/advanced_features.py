# src/advanced_features.py

import pandas as pd
import numpy as np

from pathlib import Path


# -----------------------------------
# PROJECT PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_PATH = BASE_DIR / "data" / "processed"


# -----------------------------------
# LOAD DATASETS
# -----------------------------------

matches = pd.read_csv(
    PROCESSED_PATH / "matches_cleaned.csv"
)

team_features = pd.read_csv(
    PROCESSED_PATH / "team_features.csv"
)


# -----------------------------------
# REMOVE INVALID MATCHES
# -----------------------------------

matches = matches.dropna(subset=["winner"])

matches = matches[
    matches["winner"] != "No Result"
]


# -----------------------------------
# SELECT REQUIRED MATCH COLUMNS
# -----------------------------------

matches = matches[[
    "team1",
    "team2",
    "toss_winner",
    "venue",
    "winner"
]]


# -----------------------------------
# PREPARE TEAM1 FEATURES
# -----------------------------------
team1_features = team_features.copy()

team1_features = team1_features.rename(columns={

    "team": "team1",

    "matches_played": "team1_matches_played",

    "wins": "team1_wins",

    "win_percentage": "team1_win_percentage",

    "recent_form_wins": "team1_recent_form",

    "total_runs": "team1_total_runs",

    "balls_faced": "team1_balls_faced",

    "strike_rate": "team1_strike_rate",

    "wickets": "team1_wickets",

    "balls_bowled": "team1_balls_bowled",

    "economy": "team1_economy"
})

# -----------------------------------
# PREPARE TEAM2 FEATURES
# -----------------------------------

team2_features = team_features.copy()

team2_features = team2_features.rename(columns={

    "team": "team2",

    "matches_played": "team2_matches_played",

    "wins": "team2_wins",

    "win_percentage": "team2_win_percentage",

    "recent_form_wins": "team2_recent_form",

    "total_runs": "team2_total_runs",

    "balls_faced": "team2_balls_faced",

    "strike_rate": "team2_strike_rate",

    "wickets": "team2_wickets",

    "balls_bowled": "team2_balls_bowled",

    "economy": "team2_economy"
})

# -----------------------------------
# MERGE TEAM1 FEATURES
# -----------------------------------

print("\nMerging Team1 Features...\n")

dataset = matches.merge(
    team1_features,
    on="team1",
    how="left"
)


# -----------------------------------
# MERGE TEAM2 FEATURES
# -----------------------------------

print("\nMerging Team2 Features...\n")

dataset = dataset.merge(
    team2_features,
    on="team2",
    how="left"
)


# -----------------------------------
# FEATURE DIFFERENCE CREATION
# -----------------------------------

print("\nCreating Comparative Features...\n")

dataset["win_percentage_diff"] = (
    dataset["team1_win_percentage"] -
    dataset["team2_win_percentage"]
)

dataset["recent_form_diff"] = (
    dataset["team1_recent_form"] -
    dataset["team2_recent_form"]
)

dataset["strike_rate_diff"] = (
    dataset["team1_strike_rate"] -
    dataset["team2_strike_rate"]
)

dataset["economy_diff"] = (
    dataset["team1_economy"] -
    dataset["team2_economy"]
)

dataset["wickets_diff"] = (
    dataset["team1_wickets"] -
    dataset["team2_wickets"]
)


# -----------------------------------
# HANDLE MISSING VALUES
# -----------------------------------

dataset.fillna(0, inplace=True)


# -----------------------------------
# SAVE DATASET
# -----------------------------------

save_path = (
    PROCESSED_PATH /
    "advanced_match_dataset.csv"
)

dataset.to_csv(
    save_path,
    index=False
)

print("\nAdvanced Feature Dataset Created!")

print("\nDataset Shape:", dataset.shape)

print("\nPreview:\n")

print(dataset.head())

print(f"\nDataset saved at:\n{save_path}")