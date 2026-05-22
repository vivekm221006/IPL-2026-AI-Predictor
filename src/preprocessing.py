# src/preprocessing.py

import pandas as pd
import numpy as np

from pathlib import Path
from data_loader import (
    load_matches,
    load_deliveries,
    save_processed_data
)

# -----------------------------------
# TEAM NAME NORMALIZATION
# -----------------------------------

TEAM_NAME_MAPPING = {

    # Delhi
    "Delhi Daredevils":
    "Delhi Capitals",

    # Punjab
    "Kings XI Punjab":
    "Punjab Kings",

    # Bangalore
    "Royal Challengers Bengaluru":
    "Royal Challengers Bangalore",

    # Gujarat
    "Gujarat Lions":
    "Gujarat Titans",

    # Pune
    "Rising Pune Supergiant":
    "Rising Pune Supergiants"
}


# -----------------------------------
# CLEAN MATCHES DATA
# -----------------------------------

def clean_matches_data(matches_df):

    print("\nCleaning Matches Dataset...\n")

    # Remove duplicate matches
    matches_df.drop_duplicates(
        inplace=True
    )

    # Handle missing winners
    matches_df["winner"] = matches_df[
        "winner"
    ].fillna("No Result")

    # Convert date column
    if "date" in matches_df.columns:

        matches_df["date"] = pd.to_datetime(
            matches_df["date"],
            errors="coerce"
        )

    # Standardize team names
    for old_name, new_name in TEAM_NAME_MAPPING.items():

        matches_df.replace(
            old_name,
            new_name,
            inplace=True
        )

    print("Matches dataset cleaned successfully")

    print("Shape:", matches_df.shape)

    return matches_df


# -----------------------------------
# CLEAN DELIVERIES DATA
# -----------------------------------

def clean_deliveries_data(deliveries_df):

    print("\nCleaning Deliveries Dataset...\n")

    # Remove duplicate rows
    deliveries_df.drop_duplicates(
        inplace=True
    )

    # Handle missing dismissal_kind
    deliveries_df["dismissal_kind"] = (
        deliveries_df["dismissal_kind"]
        .fillna("Not Out")
    )

    # Handle missing player_dismissed
    deliveries_df["player_dismissed"] = (
        deliveries_df["player_dismissed"]
        .fillna("None")
    )

    # Standardize team names
    for old_name, new_name in TEAM_NAME_MAPPING.items():

        deliveries_df.replace(
            old_name,
            new_name,
            inplace=True
        )

    print("Deliveries dataset cleaned successfully")

    print("Shape:", deliveries_df.shape)

    return deliveries_df


# -----------------------------------
# MAIN PREPROCESSING PIPELINE
# -----------------------------------

if __name__ == "__main__":

    print("\nStarting Preprocessing Pipeline...\n")

    # Load datasets
    matches = load_matches()

    deliveries = load_deliveries()

    # Clean datasets
    matches_cleaned = clean_matches_data(
        matches
    )

    deliveries_cleaned = clean_deliveries_data(
        deliveries
    )

    # Save processed datasets
    save_processed_data(
        matches_cleaned,
        "matches_cleaned.csv"
    )

    save_processed_data(
        deliveries_cleaned,
        "deliveries_cleaned.csv"
    )

    print("\nPreprocessing Completed Successfully!")