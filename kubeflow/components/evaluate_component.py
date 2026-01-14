from kfp.dsl import Model
import pandas as pd
import uuid
from datetime import datetime

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.metrics import RegressionMetrics
from src.utils.logger import get_logger

from src.common.snowflake_client import SnowflakeClient


# @component(
#   base_image="ml-base:latest"
# )
def evaluate_model(
    model_path,
    test_output
):
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split

    logger = get_logger(__name__)

    model = joblib.load(model_path)

    test_output_dir = Path(test_output)
    X_test = pd.read_csv(test_output_dir / "X_test.csv")
    y_test = pd.read_csv(test_output_dir / "y_test.csv")
    y_test = y_test.squeeze()

    # Evaluation
    y_pred = model.predict(X_test)
    metrics = RegressionMetrics.evaluate(y_test, y_pred)

    logger.info(f"Model evaluation: {metrics}")

    coef_df = pd.DataFrame({
        "feature": X_test.columns,
        "coefficient": model.coef_
    }).sort_values(by="coefficient", ascending=False)

    coef_df["model_name"] = "ridge_mmm_v1"
    coef_df["run_id"] = str(uuid.uuid4())
    coef_df["created_at"] = datetime.utcnow()

    # Reorder columns to match Snowflake table
    coef_df = coef_df[
        ["model_name", "feature", "coefficient", "run_id", "created_at"]
    ]

    # Write to Snowflake
    sf = SnowflakeClient()

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
        float(row["coefficient"]),     # ensure numpy → float
        row["run_id"],
        row["created_at"].isoformat()  # datetime → string
        )

        sf.execute(query, params)

    sf.close()

    logger.info("Stored MMM Coefficients to Snwoflake table 'MMM_COEFFICIENTS'")
    logger.info("Training pipeline completed")
    



if __name__ == "__main__":
    model_path = PROJECT_ROOT / "artifacts" / "model" / "ridge_mmm_model.pkl"
    test_output = PROJECT_ROOT / "artifacts" / "evaluation_data"

    evaluate_model(model_path=model_path, test_output=test_output)