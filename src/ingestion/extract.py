import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def extract_data(file_path: str) -> pd.DataFrame:
    """
    Extract marketing data from CSV file.

    Args:
        file_path (str): Path to raw CSV file

    Returns:
        pd.DataFrame: Raw data
    """
    logger.info(f"Extracting data from {file_path}")

    df = pd.read_csv(file_path)

    logger.info(f"Data extracted successfully with shape {df.shape}")
    return df