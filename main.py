from pipelines.train_pipeline import TrainPipeline
from pipelines.simulate_pipeline import SimulationPipeline
from pipelines.forecast_pipeline import ForecastPipeline

from src.ingestion.ingestion import DataIngestion
from src.models.forecasting import DemandForecaster
from src.utils.logger import logger

from pathlib import Path 

logger = logger("MAIN")

def main():
    logger.info("MMM system started")

    # -------------------------
    # Common configuration
    # -------------------------
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "synthetic_mmm_data.csv"

    channel_params = {
        "tv_spend": {"decay": 0.6, "gamma": 0.5},
        "digital_spend": {"decay": 0.4, "gamma": 0.6},
        "search_spend": {"decay": 0.3, "gamma": 0.5},
        "social_spend": {"decay": 0.5, "gamma": 0.4},
    }

    features_mmm = [
        "tv_spend_adstock",
        "digital_spend_adstock",
        "search_spend_adstock",
        "social_spend_adstock",
        "promo_flag",
        "holiday_flag",
        "price_index",
    ]

    baseline_features = [
        "price_index",
        "promo_flag",
        "holiday_flag",
        "weekofyear",
    ]

    # -------------------------
    # STEP 1 — TRAIN
    # -------------------------
    trainer = TrainPipeline(
        data_path=DATA_PATH,
        channel_params=channel_params,
        features_mmm=features_mmm,
        alpha=1.0,
    )

    model, metrics = trainer.run()
    logger.info(f"Training completed: {metrics}")

    # -------------------------
    # Load data once
    # -------------------------
    df = DataIngestion(DATA_PATH, "csv").load()

    # -------------------------
    # STEP 2 — SIMULATION
    # -------------------------
    scenarios = {
        "TV → Search (20%)": {"tv_spend": -0.2, "search_spend": 0.2},
        "Social +30%": {"social_spend": 0.3},
    }

    simulator = SimulationPipeline(
        df=df.copy(),
        channel_params=channel_params,
        features_mmm=features_mmm,
    )

    scenario_results = simulator.run(scenarios)
    logger.info("Scenario simulation completed")
    logger.info(scenario_results)

    # -------------------------
    # STEP 3 — FORECAST
    # -------------------------
    df_mmm = df.copy()

    forecaster = ForecastPipeline(
        channel_params=channel_params,
        baseline_features=baseline_features,
        features_mmm=features_mmm,
    )

    demand_forecaster = DemandForecaster(baseline_features=baseline_features, channel_params=channel_params)

    # Prepare future data (next 12 weeks) with optimized spend
    optimized_spend = {
        "social_spend": df_mmm["social_spend"].mean() * 1.3,
        "search_spend": df_mmm["search_spend"].mean() * 1.2,
        "tv_spend": df_mmm["tv_spend"].mean() * 0.8,
        "digital_spend": df_mmm["digital_spend"].mean() * 0.7
    }

    future_df = demand_forecaster.prepare_future_data(df_mmm, future_weeks=12, optimized_spend=optimized_spend)
    forecast = forecaster.run(df_mmm, future_df)

    logger.info("Forecasting completed")
    print(forecast)


if __name__ == "__main__":
    main()