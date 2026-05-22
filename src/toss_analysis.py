# src/toss_analysis.py

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
# TOSS IMPACT
# -----------------------------------

def toss_impact():

    matches["toss_match_win"] = (
        matches["toss_winner"] ==
        matches["winner"]
    )

    toss_win_rate = (
        matches["toss_match_win"]
        .mean()
    ) * 100

    return toss_win_rate


# -----------------------------------
# MAIN
# -----------------------------------

if __name__ == "__main__":

    rate = toss_impact()

    print(
        f"\nToss Winner Match Win Rate: {rate:.2f}%"
    )