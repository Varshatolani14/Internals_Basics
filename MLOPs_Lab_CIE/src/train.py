import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Paths
DATA_PATH = "data/training_data.csv"
MODEL_DIR = "models"
RESULTS_PATH = "results/step1_s1.json"
EXPERIMENT_NAME = "audittrail-audit-completion-days"


def load_data(path):
    return pd.read_csv(path)


def split_data(df):
    X = df[["controls_count", "evidence_items", "auditor_experience", "is_regulatory"]]
    y = df["audit_completion_days"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    mlflow.set_experiment(EXPERIMENT_NAME)

    models_results = []

    # -------- Linear Regression --------
    with mlflow.start_run(run_name="LinearRegression"):
        mlflow.set_tag("priority", "high")

        model = LinearRegression()

        mlflow.log_param("model_name", "LinearRegression")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae, rmse, r2 = evaluate(y_test, preds)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        path = os.path.join(MODEL_DIR, "LinearRegression.pkl")
        joblib.dump(model, path)
        mlflow.log_artifact(path)

        models_results.append({
            "name": "LinearRegression",
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        })

    # -------- Random Forest --------
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.set_tag("priority", "high")

        model = RandomForestRegressor(n_estimators=100, random_state=42)

        mlflow.log_param("model_name", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae, rmse, r2 = evaluate(y_test, preds)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        path = os.path.join(MODEL_DIR, "RandomForest.pkl")
        joblib.dump(model, path)
        mlflow.log_artifact(path)

        models_results.append({
            "name": "RandomForest",
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        })

    # -------- Select Best Model (RMSE) --------
    best_model = min(models_results, key=lambda x: x["rmse"])

    result_json = {
        "experiment_name": EXPERIMENT_NAME,
        "models": models_results,
        "best_model": best_model["name"],
        "best_metric_name": "rmse",
        "best_metric_value": float(best_model["rmse"])
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(result_json, f, indent=4)

    print("Task 1 completed. Results saved to:", RESULTS_PATH)


if __name__ == "__main__":
    main()