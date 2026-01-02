import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

NUMERIC_COLS = [
    "sales",
    "tv_spend",
    "digital_spend",
    "search_spend",
    "social_spend",
    "price_index"
]

FLAG_COLS = ["promo_flag", "holiday_flag"]


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess marketing data.

    Args:
        df (pd.DataFrame): Raw data

    Returns:
        pd.DataFrame: Cleaned data
    """
    logger.info("Starting data transformation")

    # Convert date
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

    # Sort by time (VERY IMPORTANT for MMM & forecasting)
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure numeric columns
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")

    # Ensure flags are binary ints
    df[FLAG_COLS] = df[FLAG_COLS].astype(int)

    # Handle missing values
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

    # Basic sanity checks
    if (df["sales"] < 0).any():
        raise ValueError("Negative sales detected")

    logger.info("Data transformation completed successfully")
    return df