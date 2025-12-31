import pandas as pd
from src.features.adstock import adstock_geometric
from src.features.saturation import hill_saturation


class ScenarioSimulator:
    """
    Simulates different marketing spend scenarios and computes sales lift
    """

    def __init__(self, model, df: pd.DataFrame, channel_params: dict, features: list):
        """
        Parameters
        ----------
        model : trained MMM model
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
        self.baseline_sales = self.model.predict(self.df[self.features]).sum()

    def simulate_budget_change(self, channel_changes: dict) -> float:
        """
        Simulate sales after applying budget changes
        Parameters
        ----------
        channel_changes : dict
            Keys = channel names
            Values = pct change (e.g., 0.2 for +20%, -0.2 for -20%)
        Returns
        -------
        float : simulated total sales
        """
        df_sim = self.df.copy()

        for channel, pct_change in channel_changes.items():
            df_sim[channel] *= (1 + pct_change)

            # Recompute adstock + saturation
            params = self.channel_params[channel]
            adstocked = adstock_geometric(df_sim[channel].values, decay=params.get("decay", 0.5))
            df_sim[f"{channel}_adstock"] = hill_saturation(adstocked, gamma=params.get("gamma", 0.5))

        X_sim = df_sim[self.features]
        simulated_sales = self.model.predict(X_sim).sum()

        return simulated_sales

    def scenario_lift(self, channel_changes: dict) -> float:
        """
        Returns sales lift compared to baseline
        """
        simulated_sales = self.simulate_budget_change(channel_changes)
        return simulated_sales - self.baseline_sales

    def compare_scenarios(self, scenarios: dict) -> pd.DataFrame:
        """
        Compare multiple scenarios
        Parameters
        ----------
        scenarios : dict
            Keys = scenario name
            Values = channel_changes dict
        Returns
        -------
        pd.DataFrame
        """
        data = []
        for name, changes in scenarios.items():
            simulated_sales = self.simulate_budget_change(changes)
            lift = simulated_sales - self.baseline_sales
            data.append({"Scenario": name, "Total Sales": simulated_sales, "Sales Lift": lift})

        return pd.DataFrame(data).sort_values(by="Sales Lift", ascending=False)