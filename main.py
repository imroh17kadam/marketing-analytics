from pathlib import Path

from pipelines.train_pipeline import TrainPipeline
from pipelines.simulate_pipeline import SimulationPipeline
from pipelines.forecast_pipeline import ForecastPipeline

from src.ingestion.ingestion import DataIngestion
from src.models.forecasting import DemandForecaster
from src.utils.logger import get_logger

logger = get_logger("MAIN")


QUERY = """
SELECT *
FROM MARKETING_ML.ANALYTICS.PROCESSED_MARKETING_DATA
"""

CHANNEL_PARAMS = {
    "tv_spend": {"decay": 0.6, "gamma": 0.5},
    "digital_spend": {"decay": 0.4, "gamma": 0.6},
    "search_spend": {"decay": 0.3, "gamma": 0.5},
    "social_spend": {"decay": 0.5, "gamma": 0.4},
}

FEATURES_MMM = [
    "tv_spend_adstock",
    "digital_spend_adstock",
    "search_spend_adstock",
    "social_spend_adstock",
    "promo_flag",
    "holiday_flag",
    "price_index",
]

BASELINE_FEATURES = [
    "price_index",
    "promo_flag",
    "holiday_flag",
    "weekofyear",
]


def load_data():
    """
    Centralized data loading function
    (Avoid duplicate ingestion logic)
    """
    logger.info("Loading data from Snowflake")
    return DataIngestion(
        source="file",
        query=QUERY
    ).load()


def run_training():
    logger.info("Starting Training Pipeline")

    trainer = TrainPipeline(
        query=QUERY,
        channel_params=CHANNEL_PARAMS,
        features_mmm=FEATURES_MMM,
        alpha=1.0,
    )

    model, metrics = trainer.run()
    logger.info(f"Training completed: {metrics}")

    return model, metrics


def run_simulation(df: "pd.DataFrame"):
    logger.info("Starting Simulation Pipeline")

    scenarios = {
        "TV → Search (20%)": {"tv_spend": -0.2, "search_spend": 0.2},
        "Social +30%": {"social_spend": 0.3},
    }

    simulator = SimulationPipeline(
        df=df.copy(),
        channel_params=CHANNEL_PARAMS,
        features_mmm=FEATURES_MMM,
    )

    scenario_results = simulator.run(scenarios)

    logger.info("Scenario simulation completed")
    logger.info(f"\n{scenario_results}")

    return scenario_results


def run_forecast(df: "pd.DataFrame"):
    logger.info("Starting Forecast Pipeline")

    forecaster = ForecastPipeline(
        channel_params=CHANNEL_PARAMS,
        baseline_features=BASELINE_FEATURES,
        features_mmm=FEATURES_MMM,
    )

    demand_forecaster = DemandForecaster(
        baseline_features=BASELINE_FEATURES,
        channel_params=CHANNEL_PARAMS
    )

    # Prepare optimized future spend scenario
    optimized_spend = {
        "social_spend": df["social_spend"].mean() * 1.3,
        "search_spend": df["search_spend"].mean() * 1.2,
        "tv_spend": df["tv_spend"].mean() * 0.8,
        "digital_spend": df["digital_spend"].mean() * 0.7
    }

    future_df = demand_forecaster.prepare_future_data(
        df,
        future_weeks=12,
        optimized_spend=optimized_spend
    )

    forecast = forecaster.run(df, future_df)

    logger.info("Forecasting completed")
    print(forecast)

    return forecast


def main():
    logger.info("MMM system started")

    # STEP 0 — Load data ONCE
    df = load_data()

    # STEP 1 — Train
    model, metrics = run_training()

    # STEP 2 — Simulate Scenarios
    scenario_results = run_simulation(df)

    # STEP 3 — Forecast Demand
    forecast = run_forecast(df)

    logger.info("MMM system finished successfully")


if __name__ == "__main__":
    main()