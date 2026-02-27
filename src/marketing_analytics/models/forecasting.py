# src/models/forecasting.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.marketing_analytics.features.adstock import AdstockTransformer
from src.marketing_analytics.features.saturation import SaturationTransformer
from src.marketing_analytics.utils.logger import get_logger


class DemandForecaster:
    """
    End-to-end demand forecaster combining:
    - Baseline demand model (econometric)
    - Marketing uplift from trained MMM model
    """

    def __init__(
        self,
        baseline_features: list,
        mmm_model=None,
        channel_params: dict | None = None,
        features_mmm: list | None = None,
    ):
        """
        Parameters
        ----------
        baseline_features : list
            Features for baseline demand regression

        mmm_model : trained RegularizedMMM model
            Used for marketing uplift prediction

        channel_params : dict
            Adstock + saturation parameters per channel

        features_mmm : list
            Features expected by the MMM model
        """
        if not baseline_features:
            raise ValueError("baseline_features cannot be empty.")

        self.baseline_features = baseline_features
        self.baseline_model = LinearRegression()
        self.mmm_model = mmm_model
        self.channel_params = channel_params or {}
        self.features_mmm = features_mmm or []
        self.logger = get_logger(self.__class__.__name__)

        self.logger.info("Initialized DemandForecaster")

    def _validate_columns(self, df: pd.DataFrame, required_cols: list) -> None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def _validate_numeric(self, df: pd.DataFrame, cols: list) -> None:
        non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise TypeError(f"Non-numeric columns found: {non_numeric}")

    def fit_baseline(self, df: pd.DataFrame, target_col: str = "sales"):
        """
        Train baseline demand model
        """
        self.logger.info("Training baseline demand model")

        self._validate_columns(df, self.baseline_features + [target_col])
        self._validate_numeric(df, self.baseline_features + [target_col])

        X = df[self.baseline_features]
        y = df[target_col]

        if y.isnull().any():
            raise ValueError("Target column contains NaN values.")

        self.baseline_model.fit(X, y)

        self.logger.info("Baseline model trained successfully")
        return self

    def predict_baseline(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict baseline demand
        """
        self._validate_columns(df, self.baseline_features)
        self._validate_numeric(df, self.baseline_features)

        self.logger.info(f"Generating baseline predictions for {df.shape[0]} rows")
        return self.baseline_model.predict(df[self.baseline_features])

    def compute_marketing_uplift(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict marketing uplift using trained MMM model
        """
        if self.mmm_model is None:
            raise ValueError("MMM model not provided for uplift calculation.")

        if not self.features_mmm:
            raise ValueError("features_mmm must be provided for MMM prediction.")

        self._validate_columns(df, self.features_mmm)
        self._validate_numeric(df, self.features_mmm)

        self.logger.info("Computing marketing uplift using MMM model")
        return self.mmm_model.predict(df[self.features_mmm])

    def prepare_future_data(
        self,
        df: pd.DataFrame,
        future_weeks: int,
        optimized_spend: dict,
    ) -> pd.DataFrame:
        """
        Generate future dataframe with:
        - dates
        - baseline features
        - optimized marketing spend
        - adstocked + saturated media features
        """
        self.logger.info(f"Preparing future dataset for {future_weeks} weeks")

        if future_weeks <= 0:
            raise ValueError("future_weeks must be a positive integer.")

        # Ensure date format
        df = df.copy()
        df["date"] = pd.to_datetime(
            df["date"],
            dayfirst=True,  # <-- IMPORTANT FIX
            format="mixed",  # <-- Allows mixed formats safely
            errors="raise",
        )

        last_date = df["date"].max()

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=future_weeks,
            freq="W",
        )

        future_df = pd.DataFrame(
            {
                "date": future_dates,
                "weekofyear": future_dates.isocalendar().week.astype(int),
                "price_index": 1.0,
                "promo_flag": 0,
                "holiday_flag": 0,
            }
        )

        # Add optimized spend per channel
        for channel, value in optimized_spend.items():
            future_df[channel] = value

        # Apply adstock + saturation for each channel
        for channel, params in self.channel_params.items():
            if channel not in df.columns:
                raise KeyError(f"Channel '{channel}' not found in historical data.")

            combined_series = np.concatenate(
                [
                    df[channel].values,
                    future_df[channel].values,
                ]
            )

            adstocked = AdstockTransformer.geometric(
                combined_series,
                decay=params.get("decay", 0.5),
            )[-future_weeks:]

            future_df[f"{channel}_adstock"] = SaturationTransformer.hill(
                adstocked,
                alpha=1.0,
                gamma=params.get("gamma", 0.5),
            )

        self.logger.info("Future dataset prepared successfully")
        return future_df

    def forecast(self, df_future: pd.DataFrame) -> pd.DataFrame:
        """
        Final forecast = baseline + marketing uplift
        """
        self.logger.info("Generating final demand forecast")

        baseline_pred = self.predict_baseline(df_future)
        uplift_pred = self.compute_marketing_uplift(df_future)

        df_out = df_future.copy()
        df_out["forecast_sales"] = baseline_pred + uplift_pred.values

        self.logger.info("Forecast completed successfully")
        return df_out
