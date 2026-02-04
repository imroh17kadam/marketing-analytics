import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# Import config loader from ROOT (not src)
from config_loader import ConfigLoader
from utils.logger import get_logger  # cleaner absolute import

# Load environment variables (.env)
load_dotenv()

logger = get_logger(__name__)


class DataExtractor:
    """
    Responsible for extracting raw marketing data.
    """

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize extractor.

        Args:
            file_path (Optional[str]): Path to raw CSV file.
                If None, path is loaded from config based on ENV.
        """
        self.config = ConfigLoader()

        # If file_path is not provided, use config path
        if file_path:
            self.file_path = Path(file_path)
        else:
            raw_dir = self.config.get("paths")["data_raw"]
            self.file_path = Path(raw_dir) / "marketing_data.csv"

    def validate_path(self) -> None:
        """Validate that file exists before reading."""
        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"Data file not found at {self.file_path}")

        if not self.file_path.suffix == ".csv":
            logger.error(f"Invalid file format: {self.file_path}")
            raise ValueError("Only CSV files are supported for extraction")

    def extract(self) -> pd.DataFrame:
        """
        Extract marketing data from CSV file.

        Returns:
            pd.DataFrame: Raw marketing data
        """
        logger.info(f"Extracting data from {self.file_path}")

        # Validate before reading
        self.validate_path()

        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Data extracted successfully with shape {df.shape}")
            return df

        except Exception as e:
            logger.exception(f"Failed to extract data: {e}")
            raise