import joblib
import pandas as pd

from src.marketing_analytics.features.feature_builder import MediaFeatureBuilder
from src.marketing_analytics.simulation.scenarios import ScenarioSimulator
from src.marketing_analytics.utils.logger import get_logger


class SimulationPipeline:
    """
    Pipeline for running MMM scenario simulations.
    - Loads trained MMM model
    - Applies consistent feature engineering
    - Uses ScenarioSimulator for lift computation
    - Returns ranked scenarios by sales lift
    """

    def __init__(
        self,
        df: pd.DataFrame,
        channel_params: dict,
        features_mmm: list,
        model_path: str = "artifacts/ridge_mmm_model.pkl",
    ):
        self.df = df
        self.channel_params = channel_params
        self.features_mmm = features_mmm
        self.model_path = model_path

        self.logger = get_logger(self.__class__.__name__)

        # Keep feature logic consistent with training & forecasting
        self.feature_builder = MediaFeatureBuilder(self.channel_params)

    def _load_model(self):
        """Load trained MMM model"""
        self.logger.info(f"Loading MMM model from: {self.model_path}")
        return joblib.load(self.model_path)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply consistent media feature transformation"""
        self.logger.info("Applying MediaFeatureBuilder transformation")
        return self.feature_builder.transform(df)

    def run(self, scenarios: dict) -> pd.DataFrame:
        """
        Run simulation for multiple scenarios

        Parameters
        ----------
        scenarios : dict
            {
              "scenario_name": { "tv_spend": 0.2, "digital_spend": -0.1 }
            }

        Returns
        -------
        pd.DataFrame with ranked scenarios by sales lift
        """

        self.logger.info("Simulation pipeline started")

        # ---- Load Model ----
        model = self._load_model()

        # ---- Feature Engineering (consistent with training) ----
        df_mmm = self._build_features(self.df)

        # ---- Initialize Simulator ----
        simulator = ScenarioSimulator(
            model=model,
            df=df_mmm,
            channel_params=self.channel_params,
            features=self.features_mmm,
        )

        results = []

        # ---- Run Scenarios ----
        for scenario_name, changes in scenarios.items():
            self.logger.info(f"Running scenario: {scenario_name} | Changes: {changes}")

            lift = simulator.scenario_lift(changes)

            results.append({"scenario": scenario_name, "sales_lift": lift})

        # ---- Create ranked output ----
        result_df = (
            pd.DataFrame(results)
            .sort_values(by="sales_lift", ascending=False)
            .reset_index(drop=True)
        )

        self.logger.info("Simulation pipeline completed")

        return result_df
