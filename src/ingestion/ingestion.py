import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# Project-root imports (not src.*)
from config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.common.snowflake_client import SnowflakeClient


# Load environment variables (ENV=dev/prod/test)
load_dotenv()


class DataIngestion:
    """
    Unified data ingestion layer for MMM pipelines.

    Supports:
    - File-based ingestion (csv/parquet/excel)
    - Snowflake ingestion (SQL-based)

    Automatically uses environment-based configuration.
    """

    def __init__(
        self,
        source: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        query: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        source : str
            'file' or 'snowflake'
        file_path : str, optional
            Path to file (for file source). If None, uses config default.
        file_type : str, optional
            csv | parquet | excel
        query : str, optional
            SQL query (for Snowflake)
        """

        self.source = source.lower()
        self.file_path = Path(file_path) if file_path else None
        self.file_type = file_type
        self.query = query

        # Load config based on ENV
        self.config = ConfigLoader()

        self.logger = get_logger(self.__class__.__name__)

        # Initialize Snowflake client only if needed
        self._snowflake_client = None

    def load(self) -> pd.DataFrame:
        """Main entry point for data ingestion."""

        if self.source == "file":
            return self._load_from_file()

        elif self.source == "snowflake":
            return self._load_from_snowflake()

        else:
            raise ValueError(f"Unsupported source: {self.source}. Use 'file' or 'snowflake'.")

    def _load_from_file(self) -> pd.DataFrame:
        # If no file path provided, take from config
        if self.file_path is None:
            raw_dir = self.config.get("paths")["data_raw"]
            self.file_path = Path(raw_dir) / "synthetic_mmm_data.csv"

        self.logger.info(f"Loading data from file: {self.file_path}")

        # Validate path
        self._validate_file_path(self.file_path)

        # Infer file type if not provided
        if self.file_type is None:
            self.file_type = self._infer_file_type(self.file_path)

        # Load file based on type
        if self.file_type == "csv":
            df = pd.read_csv(self.file_path)

        elif self.file_type == "parquet":
            df = pd.read_parquet(self.file_path)

        elif self.file_type in ["xls", "xlsx", "excel"]:
            df = pd.read_excel(self.file_path)

        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

        self._basic_validation(df)
        return df

    def _load_from_snowflake(self) -> pd.DataFrame:
        if not self.query:
            raise ValueError("Query must be provided for Snowflake ingestion")

        self.logger.info("Loading data from Snowflake")

        try:
            self._snowflake_client = SnowflakeClient()
            df = pd.read_sql(self.query, self._snowflake_client.conn)

            # Normalize Snowflake column names
            df.columns = df.columns.str.strip().str.lower()

            self._basic_validation(df)

            self.logger.info("Data successfully loaded from Snowflake.")
            return df

        except Exception as e:
            self.logger.exception(f"Snowflake ingestion failed: {e}")
            raise

        finally:
            if self._snowflake_client:
                self._snowflake_client.close()

    def _infer_file_type(self, path: Path) -> str:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return "csv"
        elif suffix == ".parquet":
            return "parquet"
        elif suffix in [".xls", ".xlsx"]:
            return "excel"
        else:
            raise ValueError(f"Cannot infer file type from {path}")

    def _validate_file_path(self, path: Path) -> None:
        if not path.exists():
            self.logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"Data file not found at {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

    # Basic Validation
    def _basic_validation(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")

        if df.isnull().all().any():
            self.logger.warning("Some columns contain only NULL values")