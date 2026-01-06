import pandas as pd
from typing import Optional
from src.utils.logger import get_logger
from src.common.snowflake_client import SnowflakeClient


class DataIngestion:
    """
    Unified data ingestion layer for MMM pipelines.

    Supports:
    - file-based ingestion (csv/parquet/excel)
    - Snowflake ingestion (SQL-based)
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
            Path to file (for file source)
        file_type : str, optional
            csv | parquet | excel
        query : str, optional
            SQL query (for Snowflake)
        """
        self.source = source
        self.file_path = file_path
        self.file_type = file_type
        self.query = query
        self.logger = get_logger(self.__class__.__name__)

    def load(self) -> pd.DataFrame:
        """Main entry point"""

        if self.source == "file":
            return self._load_from_file()

        elif self.source == "snowflake":
            return self._load_from_snowflake()

        else:
            raise ValueError(f"Unsupported source: {self.source}")

    # FILE INGESTION
    def _load_from_file(self) -> pd.DataFrame:
        self.logger.info(f"Loading data from file: {self.file_path}")

        if self.file_type is None:
            self.file_type = self._infer_file_type()

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

    # SNOWFLAKE INGESTION
    def _load_from_snowflake(self) -> pd.DataFrame:
        if not self.query:
            raise ValueError("Query must be provided for Snowflake ingestion")

        self.logger.info("Loading data from Snowflake")

        sf = SnowflakeClient()
        df = pd.read_sql(self.query, sf.conn)

        # Normalize Snowflake column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
        )

        sf.close()

        self._basic_validation(df)

        self.logger.info("Data Successfully Loaded from Snowflake.")
        return df

    # HELPERS
    def _infer_file_type(self) -> str:
        if self.file_path.endswith(".csv"):
            return "csv"
        elif self.file_path.endswith(".parquet"):
            return "parquet"
        elif self.file_path.endswith(".xlsx") or self.file_path.endswith(".xls"):
            return "excel"
        else:
            raise ValueError(f"Cannot infer file type from {self.file_path}")

    # Basic Validation
    def _basic_validation(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")

        if df.isnull().all().any():
            self.logger.warning("Some columns contain only NULL values")