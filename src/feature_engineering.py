# src/feature_engineering.py

import pandas as pd
import numpy as np

from pathlib import Path
from data_loader import save_processed_data


# -----------------------------------
# PROJECT PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"


# -----------------------------------
# LOAD DATA
# -----------------------------------

matches = pd.read_csv(
    PROCESSED_DATA_PATH / "matches_cleaned.csv"
)

deliveries = pd.read_csv(
    PROCESSED_DATA_PATH / "deliveries_cleaned.csv"
)


# -----------------------------------
# CREATE REQUIRED COLUMNS
# -----------------------------------

# Total runs scored on each ball
deliveries["total_runs"] = (
    deliveries["batsman_runs"] +
    deliveries["extras"]
)

# Wicket indicator
deliveries["is_wicket"] = np.where(
    deliveries["dismissal_kind"] != "Not Out",
    1,
    0
)


# -----------------------------------
# TEAM WIN PERCENTAGE
# -----------------------------------

def calculate_team_win_percentage(matches_df):

    print("\nCalculating Team Win Percentage...")

    total_matches = {}
    total_wins = {}

    teams = pd.concat([
        matches_df["team1"],
        matches_df["team2"]
    ]).unique()

    for team in teams:

        played = matches_df[
            (matches_df["team1"] == team) |
            (matches_df["team2"] == team)
        ].shape[0]

        wins = matches_df[
            matches_df["winner"] == team
        ].shape[0]

        total_matches[team] = played
        total_wins[team] = wins

    team_stats = pd.DataFrame({
        "team": teams,
        "matches_played": [
            total_matches[t] for t in teams
        ],
        "wins": [
            total_wins[t] for t in teams
        ]
    })

    team_stats["win_percentage"] = (
        team_stats["wins"] /
        team_stats["matches_played"]
    ) * 100

    return team_stats


# -----------------------------------
# RECENT FORM
# -----------------------------------

def calculate_recent_form(matches_df):

    print("\nCalculating Recent Form...")

    recent_form = {}

    teams = pd.concat([
        matches_df["team1"],
        matches_df["team2"]
    ]).unique()

    for team in teams:

        team_matches = matches_df[
            (matches_df["team1"] == team) |
            (matches_df["team2"] == team)
        ].sort_values("date")

        last_5 = team_matches.tail(5)

        wins = (
            last_5["winner"] == team
        ).sum()

        recent_form[team] = wins

    form_df = pd.DataFrame({
        "team": list(recent_form.keys()),
        "recent_form_wins": list(recent_form.values())
    })

    return form_df


# -----------------------------------
# BATTING STRENGTH
# -----------------------------------

def calculate_batting_strength(deliveries_df):

    print("\nCalculating Batting Strength...")

    batting = deliveries_df.groupby(
        "batting_team"
    ).agg({
        "total_runs": "sum",
        "ball": "count"
    }).reset_index()

    batting["strike_rate"] = (
        batting["total_runs"] /
        batting["ball"]
    ) * 100

    batting.rename(columns={
        "batting_team": "team",
        "ball": "balls_faced"
    }, inplace=True)

    return batting


# -----------------------------------
# BOWLING STRENGTH
# -----------------------------------

def calculate_bowling_strength(deliveries_df):

    print("\nCalculating Bowling Strength...")

    bowling = deliveries_df.groupby(
        "bowling_team"
    ).agg({
        "is_wicket": "sum",
        "total_runs": "sum",
        "ball": "count"
    }).reset_index()

    bowling["economy"] = (
        bowling["total_runs"] /
        (bowling["ball"] / 6)
    )

    bowling.rename(columns={
        "bowling_team": "team",
        "is_wicket": "wickets",
        "ball": "balls_bowled"
    }, inplace=True)

    return bowling


# -----------------------------------
# CREATE FINAL FEATURE DATASET
# -----------------------------------

def create_feature_dataset():

    print("\nCreating Final Feature Dataset...")

    win_df = calculate_team_win_percentage(matches)

    form_df = calculate_recent_form(matches)

    batting_df = calculate_batting_strength(deliveries)

    bowling_df = calculate_bowling_strength(deliveries)

    final_df = win_df.merge(
        form_df,
        on="team"
    )

    final_df = final_df.merge(
        batting_df,
        on="team"
    )

    final_df = final_df.merge(
        bowling_df,
        on="team"
    )

    # Remove duplicate total_runs column

    if "total_runs_y" in final_df.columns:
        final_df.drop(
            columns=["total_runs_y"],
            inplace=True
        )

    if "total_runs_x" in final_df.columns:
        final_df.rename(columns={
            "total_runs_x": "total_runs"
        }, inplace=True)

    return final_df


# -----------------------------------
# MAIN
# -----------------------------------

if __name__ == "__main__":

    features = create_feature_dataset()

    print("\nFeature Engineering Completed!")

    print("\nFeature Dataset Preview:\n")

    print(features.head())

    save_processed_data(
        features,
        "team_features.csv"
    )