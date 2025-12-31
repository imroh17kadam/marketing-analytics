import pandas as pd
from typing import Optional

from src.utils.logger import logger


class DataIngestion:
    """
    Generic data loader for MMM pipelines.

    Responsibilities:
    - Load data from disk
    - Perform basic validation
    - Keep pipeline code clean
    """

    def __init__(
        self,
        file_path: str,
        file_type: Optional[str] = None
    ):
        """
        Parameters
        ----------
        file_path : str
            Path to input data file
        file_type : str, optional
            Explicit file type: csv | parquet | excel
            If None, inferred from file extension
        """
        self.file_path = file_path
        self.file_type = file_type
        self.logger = logger(self.__class__.__name__)

    def load(self) -> pd.DataFrame:
        """Load data from disk"""

        self.logger.info(f"Loading data from {self.file_path}")

        if self.file_type is None:
            self.file_type = self._infer_file_type()

        if self.file_type == "csv":
            df = pd.read_csv(self.file_path)

        elif self.file_type == "parquet":
            df = pd.read_parquet(self.file_path)

        elif self.file_type in ["xls", "xlsx", "excel"]:
            df = pd.read_excel(self.file_path)

        else:
            raise ValueError(
                f"Unsupported file type: {self.file_type}"
            )

        self._basic_validation(df)

        self.logger.info(
            f"Data loaded successfully | Shape: {df.shape}"
        )

        return df

    def _infer_file_type(self) -> str:
        """Infer file type from extension"""

        if self.file_path.endswith(".csv"):
            return "csv"
        elif self.file_path.endswith(".parquet"):
            return "parquet"
        elif self.file_path.endswith(".xlsx") or self.file_path.endswith(".xls"):
            return "excel"
        else:
            raise ValueError(
                f"Cannot infer file type from {self.file_path}"
            )

    def _basic_validation(self, df: pd.DataFrame) -> None:
        """Basic data sanity checks"""

        if df.empty:
            raise ValueError("Loaded DataFrame is empty")

        if df.isnull().all().any():
            self.logger.warning(
                "Some columns contain only NULL values"
            )