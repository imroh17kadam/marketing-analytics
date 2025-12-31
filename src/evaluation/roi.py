import pandas as pd
from src.features.adstock import adstock_geometric
from src.features.saturation import hill_saturation


class ROIAnalyzer:
    """
    ROI and incremental sales analysis for marketing channels
    """

    def __init__(self, model, df: pd.DataFrame, channel_params: dict, features: list):
        """
        Parameters
        ----------
        model : trained MMM model (with predict method)
        df : pd.DataFrame
            Data used for predictions
        channel_params : dict
            Adstock + saturation parameters per channel
        features : list
            Features used in model
        """
        self.model = model
        self.df = df.copy()
        self.channel_params = channel_params
        self.features = features

    def incremental_sales(self, channel_cols: list) -> pd.DataFrame:
        """
        Compute incremental sales per channel using model coefficients
        """
        contributions = {}
        for col in channel_cols:
            idx = self.model.X_train.columns.get_loc(col)
            contributions[col] = self.model.model.coef_[idx] * self.df[col].sum()

        contrib_df = pd.DataFrame.from_dict(
            contributions, orient="index", columns=["incremental_sales"]
        ).sort_values(by="incremental_sales", ascending=False)

        return contrib_df

    def roi(self, channel_cols: list, raw_spend_cols: list) -> pd.DataFrame:
        """
        Compute ROI = Incremental Sales / Total Spend
        """
        contrib_df = self.incremental_sales(channel_cols)
        total_spend = self.df[raw_spend_cols].sum().values
        contrib_df["total_spend"] = total_spend
        contrib_df["ROI"] = contrib_df["incremental_sales"] / contrib_df["total_spend"]
        return contrib_df

    def simulate_roi(self, channel: str, increase_pct: float = 0.1) -> float:
        """
        Calculate ROI for a hypothetical increase in channel spend
        """
        df_copy = self.df.copy()

        # Baseline sales
        baseline_sales = self.model.predict(df_copy[self.features]).sum()

        # Increase spend
        df_copy[channel] *= (1 + increase_pct)

        # Recompute adstock and saturation
        params = self.channel_params[channel]
        adstocked = adstock_geometric(df_copy[channel].values, decay=params.get("decay", 0.5))
        df_copy[f"{channel}_adstock"] = hill_saturation(adstocked, gamma=params.get("gamma", 0.5))

        # New prediction
        new_sales = self.model.predict(df_copy[self.features]).sum()

        # ROI
        delta_sales = new_sales - baseline_sales
        delta_spend = self.df[channel].sum() * increase_pct

        return delta_sales / delta_spend

    def simulate_roi_all(self, channels: list, increase_pct: float = 0.1) -> pd.DataFrame:
        """
        Simulate ROI for all channels
        """
        roi_results = {}
        for channel in channels:
            roi_results[channel] = self.simulate_roi(channel, increase_pct)

        roi_df = pd.DataFrame.from_dict(roi_results, orient="index", columns=["ROI"])
        return roi_df