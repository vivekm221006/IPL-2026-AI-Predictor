# src/player_analysis.py

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

deliveries = pd.read_csv(
    PROCESSED_PATH /
    "deliveries_cleaned.csv"
)


# -----------------------------------
# PLAYER BATTING STATS
# -----------------------------------

def get_top_batsmen(n=10):

    batsmen = deliveries.groupby(
        "batsman"
    ).agg({

        "batsman_runs": "sum",

        "ball": "count"
    }).reset_index()

    batsmen["strike_rate"] = (
        batsmen["batsman_runs"] /
        batsmen["ball"]
    ) * 100

    batsmen = batsmen.sort_values(
        "batsman_runs",
        ascending=False
    )

    return batsmen.head(n)


# -----------------------------------
# PLAYER BOWLING STATS
# -----------------------------------

def get_top_bowlers(n=10):

    wickets = deliveries[
        deliveries["dismissal_kind"] != "Not Out"
    ]

    bowlers = wickets.groupby(
        "bowler"
    ).agg({

        "dismissal_kind": "count"
    }).reset_index()

    bowlers.rename(columns={
        "dismissal_kind": "wickets"
    }, inplace=True)

    bowlers = bowlers.sort_values(
        "wickets",
        ascending=False
    )

    return bowlers.head(n)


# -----------------------------------
# MAIN
# -----------------------------------

if __name__ == "__main__":

    print("\nTop Batsmen:\n")

    print(get_top_batsmen())

    print("\nTop Bowlers:\n")

    print(get_top_bowlers())