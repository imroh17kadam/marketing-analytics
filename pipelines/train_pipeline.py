import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from src.ingestion.ingestion import DataIngestion
from src.features.feature_builder import MediaFeatureBuilder
from src.evaluation.metrics import RegressionMetrics
from src.utils.logger import get_logger

from datetime import datetime
import uuid

from src.common.snowflake_client import SnowflakeClient

import mlflow
import mlflow.sklearn

class TrainPipeline:
    """
    End-to-end pipeline for training MMM model with MLflow tracking & registry
    """

    def __init__(
        self,
        query: str,
        channel_params: dict,
        features_mmm: list,
        target: str = "sales",
        alpha: float = 1.0,
        test_size: float = 0.2,
        experiment_name: str = "marketing-mmm"
    ):
        self.query = query
        self.channel_params = channel_params
        self.features_mmm = features_mmm
        self.target = target
        self.alpha = alpha
        self.test_size = test_size
        self.logger = get_logger(self.__class__.__name__)
        self.experiment_name = experiment_name

    def run(self):
        self.logger.info("Training pipeline started")

        # ------------------ MLFLOW SETUP ------------------
        mlflow.set_tracking_uri("http://127.0.0.1:5001")

        mlflow.set_experiment(self.experiment_name)

        run_id = str(uuid.uuid4())
        # --------------------------------------------------

        with mlflow.start_run(run_name=f"ridge-mmm-{run_id}") as run:

            # Log key parameters
            mlflow.log_param("alpha", self.alpha)
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("num_features", len(self.features_mmm))
            mlflow.set_tag("query", self.query)
            mlflow.set_tag("model_type", "Ridge-MMM")

            # Load data
            df = DataIngestion(source="snowflake", query=self.query).load()

            # Feature engineering
            builder = MediaFeatureBuilder(self.channel_params)
            df_mmm = builder.transform(df)

            # Train-test split
            X = df_mmm[self.features_mmm]
            y = df_mmm[self.target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False
            )

            # Train model
            model = Ridge(alpha=self.alpha)
            model.fit(X_train, y_train)

            # Evaluation
            y_pred = model.predict(X_test)
            metrics = RegressionMetrics.evaluate(y_test, y_pred)

            self.logger.info(f"Model evaluation: {metrics}")

            # Log metrics to MLflow
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            # --------- SAVE LOCALLY (your existing logic) ----------
            ARTIFACTS_DIR = Path("artifacts")
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

            local_model_path = ARTIFACTS_DIR / "ridge_mmm_model.pkl"
            joblib.dump(model, local_model_path)

            # --------- LOG MODEL TO MLFLOW + REGISTER -----------
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="MMM-Ridge-Model"
            )

            # --------- SAVE & LOG COEFFICIENTS ----------
            coef_df = pd.DataFrame({
                "feature": X.columns,
                "coefficient": model.coef_
            }).sort_values(by="coefficient", ascending=False)

            coef_df["model_name"] = "ridge_mmm_v1"
            coef_df["run_id"] = run.info.run_id
            coef_df["created_at"] = datetime.utcnow()

            coef_df = coef_df[
                ["model_name", "feature", "coefficient", "run_id", "created_at"]
            ]

            coef_path = ARTIFACTS_DIR / "ridge_mmm_coefficients.csv"
            coef_df.to_csv(coef_path, index=False)

            # Log coefficients as MLflow artifact
            mlflow.log_artifact(str(coef_path), artifact_path="coefficients")

            # --------- WRITE COEFFICIENTS TO SNOWFLAKE (unchanged) ----------
            sf = SnowflakeClient()

            insert_query = """
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
                    row["created_at"].isoformat()
                )
                sf.execute(insert_query, params)

            sf.close()

            self.logger.info(
                "Stored MMM Coefficients to Snowflake table 'MMM_COEFFICIENTS'"
            )

            self.logger.info(f"MLflow Run ID: {run.info.run_id}")
            self.logger.info("Training pipeline completed")

            return model, metrics