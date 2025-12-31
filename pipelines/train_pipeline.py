import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from src.ingestion.ingestion import DataIngestion
from src.features.feature_builder import MediaFeatureBuilder
from src.evaluation.metrics import RegressionMetrics
from src.utils.logger import logger


class TrainPipeline:
    """
    End-to-end pipeline for training MMM model
    """

    def __init__(
        self,
        data_path: str,
        channel_params: dict,
        features_mmm: list,
        target: str = "sales",
        alpha: float = 1.0,
        test_size: float = 0.2
    ):
        self.data_path = data_path
        self.channel_params = channel_params
        self.features_mmm = features_mmm
        self.target = target
        self.alpha = alpha
        self.test_size = test_size

        self.logger = logger(self.__class__.__name__)

    def run(self):
        self.logger.info("Training pipeline started")

        # Load data
        df = DataIngestion(self.data_path, "csv").load()

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

        coef_df.to_csv("artifacts/mmm_coefficients.csv", index=False)

        self.logger.info("Training pipeline completed")

        return model, metrics