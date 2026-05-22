# src/explainability.py

import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


# -----------------------------------
# PROJECT PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_PATH = BASE_DIR / "data" / "processed"
MODELS_PATH = BASE_DIR / "models"
OUTPUTS_PATH = BASE_DIR / "outputs" / "plots"


# -----------------------------------
# LOAD DATA
# -----------------------------------

matches = pd.read_csv(
    PROCESSED_PATH / "matches_cleaned.csv"
)

matches = matches.dropna(subset=["winner"])

matches = matches[
    matches["winner"] != "No Result"
]


# -----------------------------------
# PREPARE DATA
# -----------------------------------

dataset = matches[[
    "team1",
    "team2",
    "toss_winner",
    "venue",
    "winner"
]].copy()


# -----------------------------------
# LABEL ENCODING
# -----------------------------------

from sklearn.preprocessing import LabelEncoder

encoders = {}

for col in dataset.columns:

    le = LabelEncoder()

    dataset[col] = le.fit_transform(
        dataset[col]
    )

    encoders[col] = le


X = dataset.drop("winner", axis=1)

y = dataset["winner"]


# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = joblib.load(
    MODELS_PATH / "xgboost.pkl"
)


# -----------------------------------
# SHAP EXPLAINER
# -----------------------------------

print("\nGenerating SHAP Explainer...\n")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)


# -----------------------------------
# FEATURE IMPORTANCE PLOT
# -----------------------------------

print("\nGenerating SHAP Summary Plot...\n")

plt.figure(figsize=(10, 6))

# For multiclass models,
# choose SHAP values of first class

if isinstance(shap_values, list):

    shap_values_to_plot = shap_values[0]

else:

    shap_values_to_plot = shap_values[:, :, 0]

shap.summary_plot(
    shap_values_to_plot,
    X,
    show=False
)

plt.tight_layout()

save_path = OUTPUTS_PATH / "shap_summary.png"

plt.savefig(save_path)

print(f"\nSHAP plot saved at:\n{save_path}")