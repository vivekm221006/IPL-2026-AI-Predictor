# src/db.py

import sqlite3
import pandas as pd

from pathlib import Path


# -----------------------------------
# DATABASE PATH
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = (
    BASE_DIR /
    "data" /
    "sqlite" /
    "ipl_ai.db"
)


# -----------------------------------
# CREATE CONNECTION
# -----------------------------------

def create_connection():

    conn = sqlite3.connect(DB_PATH)

    return conn


# -----------------------------------
# SAVE DATAFRAME TO DATABASE
# -----------------------------------

def save_dataframe(df, table_name):

    conn = create_connection()

    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

    print(f"\nSaved '{table_name}' table to DB")


# -----------------------------------
# LOAD TABLE
# -----------------------------------

def load_table(table_name):

    conn = create_connection()

    query = f"SELECT * FROM {table_name}"

    df = pd.read_sql(query, conn)

    conn.close()

    return df


# -----------------------------------
# MAIN TEST
# -----------------------------------

if __name__ == "__main__":

    print("Database module ready!")