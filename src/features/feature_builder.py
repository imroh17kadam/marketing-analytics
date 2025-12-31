# src/features/feature_builder.py
import pandas as pd
from src.features.adstock import adstock_geometric
from src.features.saturation import hill_saturation


class MediaFeatureBuilder:
    """
    Builds adstocked and saturated features for marketing channels.
    """

    def __init__(self, channel_params: dict):
        """
        channel_params example:
        {
            "tv_spend": {"decay": 0.6, "gamma": 0.5},
            "digital_spend": {"decay": 0.4, "gamma": 0.6},
        }
        """
        self.channel_params = channel_params

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply adstock + saturation to all channels
        """
        df = df.copy()

        for channel, params in self.channel_params.items():
            decay = params.get("decay", 0.5)
            gamma = params.get("gamma", 0.5)

            adstocked = adstock_geometric(df[channel].values, decay=decay)
            df[f"{channel}_adstock"] = hill_saturation(adstocked, alpha=1, gamma=gamma)

        return df