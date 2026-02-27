from typing import Dict, List

import pandas as pd

# Project-relative imports (no hardcoded "src.marketing_analytics.")
from src.marketing_analytics.features.adstock import AdstockTransformer
from src.marketing_analytics.features.saturation import SaturationTransformer
from src.marketing_analytics.utils.logger import get_logger


class ScenarioSimulator:
    """
    Simulates different marketing spend scenarios and computes sales lift.
    """

    def __init__(
        self,
        model,
        df: pd.DataFrame,
        channel_params: Dict[str, dict],
        features: List[str],
    ):
        """
        Parameters
        ----------
        model : trained MMM model
            Must implement .predict()
        df : pd.DataFrame
            Original dataframe
        channel_params : dict
            Channel adstock & saturation parameters
        features : list
            Model features (including adstocked channels)
        """

        self.model = model
        self.df = df.copy()
        self.channel_params = channel_params
        self.features = features
        self.logger = get_logger(self.__class__.__name__)

        self._validate_inputs()

        # Compute baseline sales once
        self.baseline_sales = float(self.model.predict(self.df[self.features]).sum())

        self.logger.info(
            f"ScenarioSimulator initialized | baseline_sales={self.baseline_sales:.2f}"
        )

    def _validate_inputs(self) -> None:
        """Validate constructor arguments."""

        if not hasattr(self.model, "predict"):
            raise TypeError("Model must implement a predict() method")

        if not isinstance(self.df, pd.DataFrame) or self.df.empty:
            raise ValueError("Input dataframe must be a non-empty pandas DataFrame")

        if not isinstance(self.channel_params, dict) or len(self.channel_params) == 0:
            raise ValueError("channel_params must be a non-empty dictionary")

        if not isinstance(self.features, list) or len(self.features) == 0:
            raise ValueError("features must be a non-empty list")

        missing_feats = [f for f in self.features if f not in self.df.columns]
        if missing_feats:
            raise ValueError(f"Missing required features in dataframe: {missing_feats}")

    def simulate_budget_change(self, channel_changes: Dict[str, float]) -> float:
        """
        Simulate total sales after applying budget changes.

        Parameters
        ----------
        channel_changes : dict
            Keys = channel names
            Values = pct change (e.g., 0.2 for +20%, -0.2 for -20%)

        Returns
        -------
        float : simulated total sales
        """

        self.logger.info(f"Running budget simulation: {channel_changes}")

        df_sim = self.df.copy()

        for channel, pct_change in channel_changes.items():

            if channel not in df_sim.columns:
                raise KeyError(f"Channel '{channel}' not found in dataframe")

            if channel not in self.channel_params:
                raise KeyError(f"Channel '{channel}' missing in channel_params")

            if not isinstance(pct_change, (int, float)):
                raise TypeError(f"pct_change must be numeric for channel {channel}")

            # Apply spend change
            df_sim[channel] *= 1 + pct_change

            # Recompute adstock + saturation
            params = self.channel_params[channel]

            decay = params.get("decay", 0.5)
            gamma = params.get("gamma", 0.5)
            alpha = params.get("alpha", 1.0)

            self.logger.info(
                f"Recomputing transforms | channel={channel} | decay={decay} | gamma={gamma}"
            )

            adstocked = AdstockTransformer.geometric(df_sim[channel], decay=decay)

            df_sim[f"{channel}_adstock"] = SaturationTransformer.hill(
                adstocked, alpha=alpha, gamma=gamma
            )

        X_sim = df_sim[self.features]
        simulated_sales = float(self.model.predict(X_sim).sum())

        self.logger.info(
            f"Simulation completed | simulated_sales={simulated_sales:.2f}"
        )

        return simulated_sales

    def scenario_lift(self, channel_changes: Dict[str, float]) -> float:
        """
        Returns sales lift compared to baseline.
        """
        simulated_sales = self.simulate_budget_change(channel_changes)
        lift = simulated_sales - self.baseline_sales

        self.logger.info(f"Scenario lift computed | lift={lift:.2f}")

        return float(lift)

    def compare_scenarios(self, scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple scenarios.

        Parameters
        ----------
        scenarios : dict
            Keys = scenario name
            Values = channel_changes dict

        Returns
        -------
        pd.DataFrame
        """

        self.logger.info(f"Comparing {len(scenarios)} scenarios")

        results = []

        for name, changes in scenarios.items():
            self.logger.info(f"Evaluating scenario: {name}")

            simulated_sales = self.simulate_budget_change(changes)
            lift = simulated_sales - self.baseline_sales

            results.append(
                {
                    "Scenario": name,
                    "Total Sales": simulated_sales,
                    "Sales Lift": lift,
                }
            )

        result_df = (
            pd.DataFrame(results)
            .sort_values(by="Sales Lift", ascending=False)
            .reset_index(drop=True)
        )

        self.logger.info("Scenario comparison completed")

        return result_df
