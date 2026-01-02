import os
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed data to disk.

    Args:
        df (pd.DataFrame): Processed data
        output_path (str): Output CSV path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    logger.info(f"Processed data saved to {output_path}")