import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.features.adstock import adstock_geometric
from src.features.saturation import hill_saturation


class DemandForecaster:
    """
    Baseline demand forecasting with MMM integration
    """

    def __init__(self, baseline_features: list, mmm_model=None, channel_params: dict = None, features_mmm: list = None):
        """
        Parameters
        ----------
        baseline_features : list
            Features for baseline demand regression
        mmm_model : trained RegularizedMMM model
            For marketing uplift prediction
        channel_params : dict
            Adstock + saturation parameters per channel
        features_mmm : list
            Features for MMM model (adstocked channels + others)
        """
        self.baseline_features = baseline_features
        self.baseline_model = LinearRegression()
        self.mmm_model = mmm_model
        self.channel_params = channel_params
        self.features_mmm = features_mmm

    def fit_baseline(self, df: pd.DataFrame, target_col: str = "sales"):
        """
        Train baseline demand model
        """
        X = df[self.baseline_features]
        y = df[target_col]
        self.baseline_model.fit(X, y)
        return self

    def predict_baseline(self, df: pd.DataFrame):
        return self.baseline_model.predict(df[self.baseline_features])

    def compute_marketing_uplift(self, df: pd.DataFrame):
        if self.mmm_model is None:
            raise ValueError("MMM model not provided for uplift calculation.")
        return self.mmm_model.predict(df[self.features_mmm])

    def prepare_future_data(self, df: pd.DataFrame, future_weeks: int, optimized_spend: dict):
        """
        Generate future dataframe with dates, baseline features, and optimized spend
        """
        last_date = df["date"].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=future_weeks,
            freq="W"
        )

        future_df = pd.DataFrame({
            "date": future_dates,
            "weekofyear": future_dates.isocalendar().week.astype(int),
            "price_index": 1.0,
            "promo_flag": [0] * future_weeks,
            "holiday_flag": [0] * future_weeks
        })

        # Add optimized spend
        for channel, value in optimized_spend.items():
            future_df[channel] = value

        # Apply adstock + saturation for future weeks
        for channel, params in self.channel_params.items():
            adstocked = adstock_geometric(
                np.concatenate([df[channel].values, future_df[channel].values]),
                decay=params.get("decay", 0.5)
            )[-future_weeks:]

            future_df[f"{channel}_adstock"] = hill_saturation(adstocked, gamma=params.get("gamma", 0.5))

        return future_df

    def forecast(self, df_future: pd.DataFrame):
        """
        Forecast sales = baseline + marketing uplift
        """
        baseline_pred = self.predict_baseline(df_future)
        uplift_pred = self.compute_marketing_uplift(df_future)
        df_future["forecast_sales"] = baseline_pred + uplift_pred
        return df_future