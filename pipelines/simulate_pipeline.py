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

    def run(self, scenarios: list) -> pd.DataFrame:
        self.logger.info("Simulation pipeline started")

        # Load model
        model = joblib.load(self.model_path)

        # Feature engineering
        builder = MediaFeatureBuilder(self.channel_params)
        df_mmm = builder.transform(self.df)

        baseline_sales = model.predict(df_mmm[self.features_mmm]).sum()

        results = []

        for scenario in scenarios:
            simulator = ScenarioSimulator(
                model=model,
                df=df_mmm,
                channels=self.channel_params,
                features=self.features_mmm,
            )

            scenario_df = simulator.compare_scenarios(scenarios)
            print(scenario_df)

            results.append({
                "scenario": scenario["name"],
                "sales_lift": scenario_df - baseline_sales
            })

        result_df = pd.DataFrame(results).sort_values(
            by="sales_lift", ascending=False
        )

        self.logger.info("Simulation pipeline completed")

        return result_df