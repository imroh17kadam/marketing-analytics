from pipelines.train_pipeline import TrainPipeline
from pipelines.simulate_pipeline import SimulationPipeline
from pipelines.forecast_pipeline import ForecastPipeline

from src.data.data_loader import DataLoader
from src.utils.logger import get_logger

logger = get_logger("MAIN")


def main():
    logger.info("MMM system started")

    # -------------------------
    # Common configuration
    # -------------------------
    DATA_PATH = "data/processed/marketing_data.csv"

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
    df = DataLoader(DATA_PATH).load()

    # -------------------------
    # STEP 2 — SIMULATION
    # -------------------------
    scenarios = {
        "TV → Search (20%)": {"tv_spend": -0.2, "search_spend": 0.2},
        "Social +30%": {"social_spend": 0.3},
    }

    simulator = SimulationPipeline(
        df=df,
        channel_params=channel_params,
        features_mmm=features_mmm,
    )

    scenario_results = simulator.run(scenarios)
    logger.info("Scenario simulation completed")
    print(scenario_results)

    # -------------------------
    # STEP 3 — FORECAST
    # -------------------------
    forecaster = ForecastPipeline(
        channel_params=channel_params,
        baseline_features=baseline_features,
        features_mmm=features_mmm,
    )

    future_df = ...  # (your future data generator)
    forecast = forecaster.run(df, future_df)

    logger.info("Forecasting completed")
    print(forecast)


if __name__ == "__main__":
    main()