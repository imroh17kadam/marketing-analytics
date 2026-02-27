from typing import Any, Dict

import pandas as pd

from src.marketing_analytics.features.adstock import AdstockTransformer
from src.marketing_analytics.features.saturation import SaturationTransformer
from src.marketing_analytics.utils.logger import get_logger


class MediaFeatureBuilder:
    """
    Builds adstocked and saturated features for marketing channels.
    """

    def __init__(self, channel_params: Dict[str, Dict[str, Any]]):
        """
        channel_params example:
        {
            "tv_spend": {"decay": 0.6, "gamma": 0.5},
            "digital_spend": {"decay": 0.4, "gamma": 0.6},
        }
        """
        self.channel_params = channel_params
        self.logger = get_logger(self.__class__.__name__)

    def _validate_channel(self, df: pd.DataFrame, channel: str) -> None:
        """Ensure channel exists and is numeric."""
        if channel not in df.columns:
            raise KeyError(f"Channel column not found in dataframe: {channel}")

        if not pd.api.types.is_numeric_dtype(df[channel]):
            raise TypeError(
                f"Channel column must be numeric for adstock/saturation: {channel}"
            )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply adstock + saturation to all marketing channels.

        Returns:
            DataFrame with new columns:
            <channel>_adstock
        """

        self.logger.info(
            f"Building media features for channels: {list(self.channel_params.keys())}"
        )

        df_out = df.copy()

        for channel, params in self.channel_params.items():
            # Validate input column
            self._validate_channel(df_out, channel)

            decay = params.get("decay", 0.5)
            gamma = params.get("gamma", 0.5)
            alpha = params.get("alpha", 1.0)

            self.logger.info(
                f"Applying adstock + saturation for channel: {channel} "
                f"(decay={decay}, gamma={gamma}, alpha={alpha})"
            )

            # --- Your original logic preserved, just using new classes ---
            adstocked = AdstockTransformer.geometric(df_out[channel], decay=decay)

            saturated = SaturationTransformer.hill(adstocked, alpha=alpha, gamma=gamma)

            df_out[f"{channel}_adstock"] = saturated

        self.logger.info("Media feature engineering completed")
        return df_out
