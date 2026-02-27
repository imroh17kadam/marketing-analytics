import json

from kfp.dsl import Dataset, Input, Model, Output, component


@component(base_image="python:3.10", packages_to_install=["pandas", "joblib", "mlflow"])
def evaluate_model(
    test_path: Input[Dataset],
    model_artifact: Input[Model],
    evaluation_path: Output[Dataset],
):
    import uuid
    from datetime import datetime

    import joblib
    import mlflow
    import pandas as pd

    from src.marketing_analytics.common.snowflake_client import SnowflakeClient
    from src.marketing_analytics.evaluation.metrics import RegressionMetrics

    # Connect to same MLflow server
    mlflow.set_tracking_uri("http://localhost:5001")

    # Load model & test data
    model = joblib.load(model_artifact.path)
    X_test = pd.read_csv(test_path.path / "X_test.csv")
    y_test = pd.read_csv(test_path.path / "y_test.csv")

    y_pred = model.predict(X_test)
    metrics = RegressionMetrics.evaluate(y_test, y_pred)

    print("✅ Model evaluated successfully.")

    # Save metrics locally
    metrics_path = evaluation_path.path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Log metrics to MLflow
    with mlflow.start_run(run_name="kubeflow_ridge_evaluation"):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        print("✅ Metrics logged to MLflow")

    # Store coefficients in Snowflake (same as before)
    coef_df = pd.DataFrame(
        {"feature": X_test.columns, "coefficient": model.coef_}
    ).sort_values(by="coefficient", ascending=False)

    sf = SnowflakeClient()

    coef_df["model_name"] = "ridge_mmm_v1"
    coef_df["run_id"] = str(uuid.uuid4())
    coef_df["created_at"] = datetime.utcnow()

    query = """
    INSERT INTO MMM_COEFFICIENTS (
        model_name,
        feature,
        coefficient,
        run_id,
        created_at
    )
    VALUES (%s, %s, %s, %s, %s)
    """

    for _, row in coef_df.iterrows():
        params = (
            row["model_name"],
            row["feature"],
            float(row["coefficient"]),
            row["run_id"],
            row["created_at"].isoformat(),
        )
        sf.execute(query, params)

    sf.close()

    print("✅ Coefficients saved to Snowflake")
    print("✅ Evaluation completed.")
