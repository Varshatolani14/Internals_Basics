import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Paths
TRAIN_PATH = "data/training_data.csv"
NEW_PATH = "data/new_data.csv"
MODEL_DIR = "models"
RESULT_PATH = "results/step4_s8.json"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    new_df = pd.read_csv(NEW_PATH)
    return train_df, new_df


def prepare(df):
    X = df[["controls_count", "evidence_items", "auditor_experience", "is_regulatory"]]
    y = df["audit_completion_days"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    os.makedirs("results", exist_ok=True)

    train_df, new_df = load_data()

    combined_df = pd.concat([train_df, new_df], ignore_index=True)

    # -------- Load champion model (best from Task 1) --------
    lr_path = os.path.join(MODEL_DIR, "LinearRegression.pkl")
    rf_path = os.path.join(MODEL_DIR, "RandomForest.pkl")

    # Based on your results → LinearRegression was best
    champion_model = joblib.load(lr_path)

    # -------- Evaluate champion --------
    X_train, X_test, y_train, y_test = prepare(train_df)

    champion_preds = champion_model.predict(X_test)
    champion_rmse = rmse(y_test, champion_preds)

    # -------- Retrain on combined data --------
    X_train_new, X_test_new, y_train_new, y_test_new = prepare(combined_df)

    retrained_model = LinearRegression()  # same as best model
    retrained_model.fit(X_train_new, y_train_new)

    retrained_preds = retrained_model.predict(X_test_new)
    retrained_rmse = rmse(y_test_new, retrained_preds)

    improvement = champion_rmse - retrained_rmse

    if retrained_rmse < champion_rmse:
        action = "promoted"
    else:
        action = "kept_champion"

    result = {
        "original_data_rows": int(len(train_df)),
        "new_data_rows": int(len(new_df)),
        "combined_data_rows": int(len(combined_df)),
        "champion_rmse": float(champion_rmse),
        "retrained_rmse": float(retrained_rmse),
        "improvement": float(improvement),
        "min_improvement_threshold": 0,
        "action": action,
        "comparison_metric": "rmse"
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=4)

    print("Task 4 completed. Result saved to:", RESULT_PATH)


if __name__ == "__main__":
    main()