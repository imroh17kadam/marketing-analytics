from kfp.dsl import component, Input, Dataset, Model
from pathlib import Path
import json

# @component(base_image="python:3.10")
def evaluate_model(
    input_path: str | Path,
    model_artifact: str | Path,
    evaluation_path: str | Path
):
    """
    Store MMM coefficients into Snowflake.
    """
    import pandas as pd
    import joblib
    import uuid
    from datetime import datetime
    from src.common.snowflake_client import SnowflakeClient
    from src.evaluation.metrics import RegressionMetrics

    input_path = Path(input_path)
    model_artifact = Path(model_artifact)
    evaluation_path = Path(evaluation_path)

    evaluation_path.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_artifact)
    X_test = pd.read_csv(input_path / "X_test.csv")
    y_test = pd.read_csv(input_path / "y_test.csv")


    y_pred = model.predict(X_test)
    metrics = RegressionMetrics.evaluate(y_test, y_pred)
    print(f"✅ Model evaluated successfully.")

    metrics_path = evaluation_path / "metrics.json"
    print(f"✅ Metrics saved to {metrics_path}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    coef_df = pd.DataFrame({
        "feature": X_test.columns,
        "coefficient": model.coef_
    }).sort_values(by="coefficient", ascending=False)

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
            row["created_at"].isoformat()  # 🔑 FIX
        )
        sf.execute(query, params)

    sf.close()

    print(f"✅ Coefficients saved to Snowflake")
    print(f"✅ Pipelines successfully completed.")