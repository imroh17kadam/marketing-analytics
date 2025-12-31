import pandas as pd
import numpy as np


class Preprocessor:
    """
    Data preprocessing for MMM
    """

    def __init__(self):
        pass

    @staticmethod
    def fill_missing(df: pd.DataFrame, strategy="mean") -> pd.DataFrame:
        """
        Fill missing values
        """
        df_filled = df.copy()
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if strategy == "mean":
                    df_filled[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "median":
                    df_filled[col].fillna(df[col].median(), inplace=True)
                elif strategy == "zero":
                    df_filled[col].fillna(0, inplace=True)
        return df_filled

    @staticmethod
    def encode_flags(df: pd.DataFrame, flag_columns: list) -> pd.DataFrame:
        """
        Ensure promo/holiday flags are 0/1
        """
        df_encoded = df.copy()
        for col in flag_columns:
            df_encoded[col] = df_encoded[col].astype(int)
        return df_encoded

    @staticmethod
    def log_transform(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        Apply log transformation to numeric columns
        """
        df_log = df.copy()
        for col in cols:
            df_log[col] = np.log1p(df_log[col])
        return df_log