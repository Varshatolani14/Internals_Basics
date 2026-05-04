import mlflow
from mlflow.tracking import MlflowClient
import json
import os

MODEL_NAME = "audittrail-audit-completion-days-predictor"
EXPERIMENT_NAME = "audittrail-audit-completion-days"


def main():
    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise Exception("Experiment not found")

    # Get all runs
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    # Sort by RMSE (lower is better)
    best_run = sorted(runs, key=lambda x: float(x.data.metrics["rmse"]))[0]

    run_id = best_run.info.run_id
    rmse = best_run.data.metrics["rmse"]

    # Model URI
    model_uri = f"runs:/{run_id}/model"

    # Register model
    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("Registered model version:", result.version)
    print("Run ID:", run_id)

    # Save JSON output
    os.makedirs("results", exist_ok=True)

    output = {
        "registered_model_name": MODEL_NAME,
        "version": int(result.version),
        "run_id": run_id,
        "source_metric": "rmse",
        "source_metric_value": float(rmse)
    }

    with open("results/step3_s6.json", "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()