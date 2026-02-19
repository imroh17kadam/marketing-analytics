import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

from src.features.feature_builder import MediaFeatureBuilder
from src.utils.logger import get_logger


class ForecastPipeline:
    """
    Demand forecasting pipeline with MMM uplift.
    - Trains a baseline demand model
    - Loads trained MMM model
    - Applies consistent feature transformation
    - Combines baseline + uplift forecast
    """

    def __init__(
        self,
        channel_params: dict,
        baseline_features: list,
        features_mmm: list,
        model_path: str = "artifacts/ridge_mmm_model.pkl"
    ):
        self.channel_params = channel_params
        self.baseline_features = baseline_features
        self.features_mmm = features_mmm
        self.model_path = model_path

        self.logger = get_logger(self.__class__.__name__)

        # Feature builder (consistent with training)
        self.feature_builder = MediaFeatureBuilder(self.channel_params)

    def _prepare_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Standardizing date format")

        df = df.copy()

        # Robust date parsing (handles your DD-MM-YYYY format)
        df["date"] = pd.to_datetime(
            df["date"],
            dayfirst=True,
            format="mixed",
            errors="raise",
        )

        # Sanity check — fail early if anything went wrong
        if df["date"].isna().any():
            raise ValueError(
                "Some dates could not be parsed. Please check 'date' column format."
            )

        # Add time features
        df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

        self.logger.info("Date preparation completed")
        return df

    def _train_baseline_model(self, historical_df: pd.DataFrame) -> LinearRegression:
        """Train baseline demand model"""
        self.logger.info("Training baseline demand model")

        model = LinearRegression()
        model.fit(
            historical_df[self.baseline_features],
            historical_df["sales"]
        )

        return model

    def run(
        self,
        historical_df: pd.DataFrame,
        future_df: pd.DataFrame
    ) -> pd.DataFrame:

        self.logger.info("Forecast pipeline started")

        # ---- Load trained MMM model ----
        self.logger.info(f"Loading MMM model from: {self.model_path}")
        mmm_model = joblib.load(self.model_path)

        # ---- Prepare data ----
        historical_df = self._prepare_dates(historical_df)
        future_df = self._prepare_dates(future_df)

        # ---- Train baseline model ----
        baseline_model = self._train_baseline_model(historical_df)

        # ---- Apply SAME feature transformation as training ----
        self.logger.info("Applying media feature transformation to future data")

        # Important: We transform future_df so it has *_adstock features
        future_transformed = self.feature_builder.transform(future_df)

        # ---- Forecast ----
        self.logger.info("Generating forecasts")

        future_base = baseline_model.predict(
            future_df[self.baseline_features]
        )

        future_uplift = mmm_model.predict(
            future_transformed[self.features_mmm]
        )

        future_df["forecast_sales"] = future_base + future_uplift

        self.logger.info("Forecast pipeline completed")

        return future_df[["date", "forecast_sales"]]