from typing import List

import pandas as pd

# Use project-level imports (not hardcoded paths)
from simulation.scenarios import ScenarioSimulator

from src.marketing_analytics.utils.logger import get_logger


class BudgetOptimizer:
    """
    Automatically evaluates which channel gives highest lift
    for a fixed percentage increase in spend.
    """

    def __init__(
        self,
        simulator: ScenarioSimulator,
        channels: List[str],
        increase_pct: float = 0.2,
    ):
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
        self.logger = get_logger(self.__class__.__name__)

        self._validate_inputs()

        self.logger.info(
            f"Initialized BudgetOptimizer | channels={channels} | increase_pct={increase_pct}"
        )

    def _validate_inputs(self) -> None:
        """Validate constructor inputs."""

        if not isinstance(self.simulator, ScenarioSimulator):
            raise TypeError(
                f"Expected simulator of type ScenarioSimulator, got {type(self.simulator)}"
            )

        if not isinstance(self.channels, list) or len(self.channels) == 0:
            raise ValueError("channels must be a non-empty list of strings")

        if not all(isinstance(c, str) for c in self.channels):
            raise TypeError("All channel names must be strings")

        if not (0 < self.increase_pct < 1):
            raise ValueError(
                f"increase_pct must be between 0 and 1 (exclusive). Got: {self.increase_pct}"
            )

    def optimize(self) -> pd.DataFrame:
        """
        Evaluate expected sales lift for each channel.

        Returns
        -------
        pd.DataFrame
            Columns: ['channel', 'sales_lift']
            Sorted by descending sales lift
        """

        self.logger.info("Starting budget optimization simulation")

        results = []

        for channel in self.channels:
            self.logger.info(
                f"Simulating lift for channel={channel} with increase_pct={self.increase_pct}"
            )

            lift = self.simulator.scenario_lift({channel: self.increase_pct})

            results.append(
                {
                    "channel": channel,
                    "sales_lift": float(lift),
                }
            )

        result_df = (
            pd.DataFrame(results)
            .sort_values(by="sales_lift", ascending=False)
            .reset_index(drop=True)
        )

        self.logger.info("Budget optimization completed successfully")

        return result_df
