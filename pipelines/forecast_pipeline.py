import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

from src.features.adstock import adstock_geometric
from src.features.saturation import hill_saturation
from src.utils.logger import logger


class ForecastPipeline:
    """
    Demand forecasting pipeline with MMM uplift
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

        self.logger = logger(self.__class__.__name__)

    def run(
        self,
        historical_df: pd.DataFrame,
        future_df: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.info("Forecast pipeline started")

        # Load MMM model
        mmm_model = joblib.load(self.model_path)

        historical_df["date"] = pd.to_datetime(historical_df["date"], dayfirst=True)
        historical_df["weekofyear"] = historical_df["date"].dt.isocalendar().week

        # Train baseline demand model
        baseline_model = LinearRegression()
        baseline_model.fit(
            historical_df[self.baseline_features],
            historical_df["sales"]
        )

        # Apply adstock to future data
        # for channel, params in self.channel_params.items():
        #     adstocked = adstock_geometric(
        #         np.concatenate([historical_df[channel].values, future_df[channel].values]),
        #         decay=params["decay"]
        #     )[-len(future_df):]

        #     future_df[f"{channel}_adstock"] = hill_saturation(adstocked, gamma=params["gamma"])

        # Forecast
        future_base = baseline_model.predict(
            future_df[self.baseline_features]
        )
        future_uplift = mmm_model.predict(
            future_df[self.features_mmm]
        )

        future_df["forecast_sales"] = future_base + future_uplift

        self.logger.info("Forecast pipeline completed")

        return future_df[["date", "forecast_sales"]]