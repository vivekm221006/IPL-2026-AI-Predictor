# src/utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# -----------------------------------
# MODEL EVALUATION
# -----------------------------------

def evaluate_model(
    y_true,
    y_pred,
    model_name="Model"
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print(f"\n{model_name} Accuracy:")
    print(f"{accuracy:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred
        )
    )

    return accuracy


# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    title="Confusion Matrix"
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(cm)

    plt.title(title)

    plt.colorbar()

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()


# -----------------------------------
# SAVE CSV
# -----------------------------------

def save_csv(df, path):

    df.to_csv(
        path,
        index=False
    )

    print(f"\nCSV saved at:\n{path}")


# -----------------------------------
# RANDOM SEED
# -----------------------------------

def set_random_seed(seed=42):

    np.random.seed(seed)

    print(f"\nRandom seed set to {seed}")