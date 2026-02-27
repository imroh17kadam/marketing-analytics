from typing import List

import numpy as np
import pandas as pd

from src.marketing_analytics.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Centralized data preprocessing for MMM pipelines.

    Responsibilities:
    - Handle missing values
    - Encode binary flag columns
    - Apply safe log transformations
    """

    logger = get_logger(__qualname__)

    @staticmethod
    def fill_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        Fill missing values using specified strategy.

        Args:
            df: Input dataframe
            strategy: 'mean' | 'median' | 'zero'

        Returns:
            A new dataframe with missing values handled
        """

        Preprocessor.logger.info(f"Filling missing values using strategy: {strategy}")

        if strategy not in ["mean", "median", "zero"]:
            raise ValueError(
                f"Invalid strategy: {strategy}. Choose from ['mean', 'median', 'zero']"
            )

        df_filled = df.copy()

        for col in df_filled.columns:
            if df_filled[col].isnull().sum() > 0:

                if strategy == "mean":
                    value = df_filled[col].mean()

                elif strategy == "median":
                    value = df_filled[col].median()

                else:  # zero
                    value = 0

                df_filled[col] = df_filled[col].fillna(value)

        Preprocessor.logger.info("Missing values filled successfully")
        return df_filled

    @staticmethod
    def encode_flags(df: pd.DataFrame, flag_columns: List[str]) -> pd.DataFrame:
        """
        Ensure promo/holiday flags are 0/1 integers.

        Args:
            df: Input dataframe
            flag_columns: List of binary flag column names

        Returns:
            DataFrame with encoded flags
        """

        Preprocessor.logger.info(f"Encoding flag columns: {flag_columns}")

        df_encoded = df.copy()

        for col in flag_columns:

            if col not in df_encoded.columns:
                raise KeyError(f"Flag column not found in dataframe: {col}")

            # Ensure safe conversion
            df_encoded[col] = df_encoded[col].fillna(0).astype(int)

        Preprocessor.logger.info("Flag encoding completed")
        return df_encoded

    @staticmethod
    def log_transform(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Apply log(1 + x) transformation to specified numeric columns.

        Args:
            df: Input dataframe
            cols: List of numeric columns to transform

        Returns:
            Transformed dataframe
        """

        Preprocessor.logger.info(f"Applying log transform to columns: {cols}")

        df_log = df.copy()

        for col in cols:

            if col not in df_log.columns:
                raise KeyError(f"Column not found for log transform: {col}")

            if not np.issubdtype(df_log[col].dtype, np.number):
                raise TypeError(f"Column must be numeric for log transform: {col}")

            # Avoid negative values issue
            if (df_log[col] < 0).any():
                Preprocessor.logger.warning(
                    f"Column {col} contains negative values — shifting before log transform"
                )
                min_val = df_log[col].min()
                df_log[col] = df_log[col] - min_val + 1

            df_log[col] = np.log1p(df_log[col])

        Preprocessor.logger.info("Log transformation completed")
        return df_log
