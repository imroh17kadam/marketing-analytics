import pandas as pd
import numpy as np

from src.features.adstock import AdstockTransformer
from src.features.saturation import SaturationTransformer

from utils.logger import get_logger


class ROIAnalyzer:
    """
    ROI and incremental sales analysis for marketing channels.
    This version is fully aligned with your refactored feature transformers
    and MMM model design.
    """

    def __init__(
        self,
        model,
        df: pd.DataFrame,
        channel_params: dict,
        features_mmm: list,
        raw_spend_cols: list,
    ):
        """
        Parameters
        ----------
        model : trained RegularizedMMM
        df : pd.DataFrame
            Historical dataset
        channel_params : dict
            Adstock + saturation parameters per channel
        features_mmm : list
            Final feature set used in MMM model (including *_adstock)
        raw_spend_cols : list
            Original spend columns (e.g. ["tv_spend", "digital_spend"])
        """
        self.model = model
        self.df = df.copy()
        self.channel_params = channel_params
        self.features_mmm = features_mmm
        self.raw_spend_cols = raw_spend_cols
        self.logger = get_logger(self.__class__.__name__)

    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recompute adstock + saturation for all channels in a consistent way.
        This avoids duplication across methods.
        """
        df = df.copy()

        for channel, params in self.channel_params.items():
            decay = params.get("decay", 0.5)
            gamma = params.get("gamma", 0.5)
            alpha = params.get("alpha", 1.0)

            # Adstock
            adstocked = AdstockTransformer.geometric(
                df[channel].values, decay=decay
            )

            # Saturation
            df[f"{channel}_adstock"] = SaturationTransformer.hill(
                adstocked, alpha=alpha, gamma=gamma
            )

        return df

    def _predict_total_sales(self, df: pd.DataFrame) -> float:
        """
        Predict total sales using the MMM model.
        Centralized to avoid repetition.
        """
        return float(self.model.predict(df[self.features_mmm]).sum())

    def incremental_sales(self, channel_cols: list) -> pd.DataFrame:
        """
        Compute incremental sales per channel using model coefficients
        (cleaner, safer implementation).
        """
        coef_df = self.model.get_coefficients().set_index("feature")

        contributions = {}

        for col in channel_cols:
            if col not in coef_df.index:
                raise ValueError(f"Feature {col} not found in model coefficients.")

            coef = coef_df.loc[col, "coefficient"]
            total_feature_value = self.df[col].sum()

            contributions[col] = coef * total_feature_value

        self.logger.info(f"Computing incremental sales for channels: {channel_cols}")

        contrib_df = (
            pd.DataFrame.from_dict(
                contributions, orient="index", columns=["incremental_sales"]
            )
            .sort_values(by="incremental_sales", ascending=False)
        )

        return contrib_df

    def roi(self, channel_cols: list) -> pd.DataFrame:
        """
        Compute ROI = Incremental Sales / Total Spend
        """
        contrib_df = self.incremental_sales(channel_cols)

        total_spend = self.df[self.raw_spend_cols].sum()

        # Attach spend to corresponding channels
        contrib_df["total_spend"] = total_spend.loc[contrib_df.index]

        contrib_df["ROI"] = (
            contrib_df["incremental_sales"] / contrib_df["total_spend"]
        )

        return contrib_df

    def simulate_roi(self, channel: str, increase_pct: float = 0.1) -> float:
        """
        Calculate ROI for a hypothetical increase in channel spend.
        Properly recomputes transformations.
        """

        if channel not in self.channel_params:
            raise ValueError(f"Channel {channel} not in channel_params.")

        # 1) Baseline prediction
        df_base = self._apply_transformations(self.df)
        baseline_sales = self._predict_total_sales(df_base)

        # 2) Increase spend
        df_sim = self.df.copy()
        df_sim[channel] *= 1 + increase_pct

        # 3) Recompute adstock + saturation
        df_sim = self._apply_transformations(df_sim)

        # 4) New prediction
        new_sales = self._predict_total_sales(df_sim)

        # 5) Compute ROI
        delta_sales = new_sales - baseline_sales
        delta_spend = self.df[channel].sum() * increase_pct

        return float(delta_sales / delta_spend)

    def simulate_roi_all(
        self, channels: list, increase_pct: float = 0.1
    ) -> pd.DataFrame:
        """
        Simulate ROI for all channels in one shot.
        """

        results = {
            channel: self.simulate_roi(channel, increase_pct)
            for channel in channels
        }

        return pd.DataFrame.from_dict(
            results, orient="index", columns=["ROI"]
        ).sort_values(by="ROI", ascending=False)