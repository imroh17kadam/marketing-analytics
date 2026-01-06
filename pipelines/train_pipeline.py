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

class TrainPipeline:
    """
    End-to-end pipeline for training MMM model
    """

    def __init__(
        self,
        query: str,
        # data_path: str,
        channel_params: dict,
        features_mmm: list,
        target: str = "sales",
        alpha: float = 1.0,
        test_size: float = 0.2
    ):
        self.query = query
        # self.data_path = data_path
        self.channel_params = channel_params
        self.features_mmm = features_mmm
        self.target = target
        self.alpha = alpha
        self.test_size = test_size
        self.logger = get_logger(self.__class__.__name__)

    def run(self):
        self.logger.info("Training pipeline started")

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

        ARTIFACTS_DIR = Path("artifacts")
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = ARTIFACTS_DIR / "ridge_mmm_model.pkl"
        joblib.dump(model, model_path)

        # Save artifacts
        joblib.dump(model, "artifacts/ridge_mmm_model.pkl")

        coef_df = pd.DataFrame({
            "feature": X.columns,
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

        self.logger.info("Stored MMM Coefficients to Snwoflake table 'MMM_COEFFICIENTS'")
        
        self.logger.info("Training pipeline completed")

        return model, metrics