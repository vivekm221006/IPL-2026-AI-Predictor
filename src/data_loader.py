# src/data_loader.py

import pandas as pd
import sqlite3
from pathlib import Path


# Root Project Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"
SQLITE_PATH = BASE_DIR / "data" / "sqlite" / "ipl.db"


def load_matches():
    """
    Load IPL matches dataset
    """

    file_path = RAW_DATA_PATH / "matches.csv"

    try:
        matches = pd.read_csv(file_path)

        print("Matches Dataset Loaded Successfully")
        print("Shape:", matches.shape)

        return matches

    except FileNotFoundError:
        print("ERROR: matches.csv not found")
        return None


def load_deliveries():
    """
    Load IPL deliveries dataset
    """

    file_path = RAW_DATA_PATH / "deliveries.csv"

    try:
        deliveries = pd.read_csv(file_path)

        print("Deliveries Dataset Loaded Successfully")
        print("Shape:", deliveries.shape)

        return deliveries

    except FileNotFoundError:
        print("ERROR: deliveries.csv not found")
        return None


def save_processed_data(df, filename):
    """
    Save processed dataframe
    """

    save_path = PROCESSED_DATA_PATH / filename

    df.to_csv(save_path, index=False)

    print(f"Processed data saved at: {save_path}")


def connect_database():
    """
    Connect SQLite database
    """

    conn = sqlite3.connect(SQLITE_PATH)

    print("Connected to SQLite Database")

    return conn


def save_to_database(df, table_name):
    """
    Save dataframe into SQLite table
    """

    conn = connect_database()

    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False
    )

    conn.commit()
    conn.close()

    print(f"Saved table: {table_name}")


if __name__ == "__main__":

    print("\nLoading Datasets...\n")

    matches = load_matches()
    deliveries = load_deliveries()

    if matches is not None:
        print("\nMatches Preview:\n")
        print(matches.head())

    if deliveries is not None:
        print("\nDeliveries Preview:\n")
        print(deliveries.head())