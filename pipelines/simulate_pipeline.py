import joblib
import pandas as pd

from src.features.feature_builder import MediaFeatureBuilder
from src.simulation.scenarios import ScenarioSimulator
from src.utils.logger import get_logger


class SimulationPipeline:
    """
    Pipeline for running MMM scenario simulations
    """

    def __init__(
        self,
        df: pd.DataFrame,
        channel_params: dict,
        features_mmm: list,
        model_path: str = "artifacts/ridge_mmm_model.pkl"
    ):
        self.df = df
        self.channel_params = channel_params
        self.features_mmm = features_mmm
        self.model_path = model_path

        self.logger = get_logger(self.__class__.__name__)

    def run(self, scenarios: dict) -> pd.DataFrame:
        self.logger.info("Simulation pipeline started")

        # Load model
        model = joblib.load(self.model_path)

        # Feature engineering
        builder = MediaFeatureBuilder(self.channel_params)
        df_mmm = builder.transform(self.df)

        baseline_sales = model.predict(df_mmm[self.features_mmm]).sum()

        results = []

        simulator = ScenarioSimulator(
            model=model,
            df=df_mmm,
            channel_params=self.channel_params,
            features=self.features_mmm,
        )

        for scenario_name, changes in scenarios.items():
            scenario_dict = {scenario_name: changes}
            scenario_df = simulator.compare_scenarios(scenario_dict)

            self.logger.info(f"scenario_dict: {scenario_dict}")
            self.logger.info(f"scenario_df: {scenario_df}")

            for _, row in scenario_df.iterrows():
                results.append({
                    "scenario": row["Scenario"],
                    "sales_lift": row["Sales Lift"]
                })

        result_df = pd.DataFrame(results).sort_values(
            by="sales_lift", ascending=False
        )

        self.logger.info("Simulation pipeline completed")

        return result_df