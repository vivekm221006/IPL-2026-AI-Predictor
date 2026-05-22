# src/venue_analysis.py

import pandas as pd

from pathlib import Path


# -----------------------------------
# PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_PATH = (
    BASE_DIR /
    "data" /
    "processed"
)


# -----------------------------------
# LOAD DATA
# -----------------------------------

matches = pd.read_csv(
    PROCESSED_PATH /
    "matches_cleaned.csv"
)


# -----------------------------------
# VENUE STATS
# -----------------------------------

def venue_win_stats():

    venue_stats = matches.groupby(
        ["venue", "winner"]
    ).size().reset_index(name="wins")

    return venue_stats


# -----------------------------------
# AVERAGE SCORE PER VENUE
# -----------------------------------

def average_scores():

    if "target_runs" not in matches.columns:

        return pd.DataFrame()

    scores = matches.groupby(
        "venue"
    )["target_runs"].mean().reset_index()

    scores.rename(columns={
        "target_runs": "avg_score"
    }, inplace=True)

    return scores


# -----------------------------------
# MAIN
# -----------------------------------

if __name__ == "__main__":

    print("\nVenue Win Stats:\n")

    print(venue_win_stats().head())