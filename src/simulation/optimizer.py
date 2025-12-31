import pandas as pd
from src.simulation.scenarios import ScenarioSimulator


class BudgetOptimizer:
    """
    Automatically evaluates which channel gives highest lift for a fixed % increase
    """

    def __init__(self, simulator: ScenarioSimulator, channels: list, increase_pct: float = 0.2):
        """
        Parameters
        ----------
        simulator : ScenarioSimulator
            Instance of scenario simulator
        channels : list
            List of media channels to test
        increase_pct : float
            Percent increase to simulate (default 20%)
        """
        self.simulator = simulator
        self.channels = channels
        self.increase_pct = increase_pct

    def optimize(self) -> pd.DataFrame:
        """
        Returns DataFrame with expected lift for each channel
        """
        results = []
        for channel in self.channels:
            lift = self.simulator.scenario_lift({channel: self.increase_pct})
            results.append({"channel": channel, "sales_lift": lift})

        return pd.DataFrame(results).sort_values(by="sales_lift", ascending=False)