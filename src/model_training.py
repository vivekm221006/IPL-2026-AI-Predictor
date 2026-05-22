# src/model_training.py

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# -----------------------------------
# PROJECT PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_PATH = BASE_DIR / "data" / "processed"
MODELS_PATH = BASE_DIR / "models"


# -----------------------------------
# LOAD ADVANCED DATASET
# -----------------------------------

dataset = pd.read_csv(
    PROCESSED_PATH /
    "advanced_match_dataset.csv"
)


# -----------------------------------
# ENCODE CATEGORICAL FEATURES
# -----------------------------------

print("\nEncoding Categorical Features...\n")

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
# FEATURES & TARGET
# -----------------------------------

X = dataset.drop("winner", axis=1)

y = dataset["winner"]


# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# MODELS
# -----------------------------------

models = {

    "random_forest":
    RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    ),

    "extra_trees":
    ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    ),

    "xgboost":
    XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),

    "lightgbm":
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=8,
        random_state=42
    )
}


# -----------------------------------
# TRAINING LOOP
# -----------------------------------

results = {}

for name, model in models.items():

    print(f"\nTraining {name}...\n")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy:.4f}")

    # Save model
    joblib.dump(
        model,
        MODELS_PATH / f"{name}.pkl"
    )


# -----------------------------------
# BEST MODEL
# -----------------------------------

best_model = max(
    results,
    key=results.get
)

print("\n================================")
print("BEST MODEL:", best_model)
print("BEST ACCURACY:", results[best_model])
print("================================")


print("\nAll Models Trained Successfully!")