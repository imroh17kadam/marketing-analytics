from typing import Union

import numpy as np
import pandas as pd

from src.marketing_analytics.utils.logger import get_logger

logger = get_logger(__name__)


class AdstockTransformer:
    """
    Implements different adstock transformations for MMM.

    Currently supports:
    - Geometric adstock (default)
    """

    logger = get_logger(__qualname__)

    @staticmethod
    def validate_decay(decay: float) -> None:
        if not (0 < decay < 1):
            raise ValueError(f"Decay must be between 0 and 1 (exclusive). Got: {decay}")

    @staticmethod
    def validate_series(series: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Ensure valid numeric input and convert to numpy array.
        """
        if isinstance(series, pd.Series):
            series = series.to_numpy()

        if not isinstance(series, np.ndarray):
            raise TypeError(
                f"Expected numpy array or pandas Series, got {type(series)}"
            )

        if np.isnan(series).any():
            AdstockTransformer.logger.warning(
                "Input series contains NaN values — replacing with 0"
            )
            series = np.nan_to_num(series, nan=0.0)

        if (series < 0).any():
            AdstockTransformer.logger.warning(
                "Input series contains negative values — setting them to 0 for adstock"
            )
            series = np.clip(series, a_min=0, a_max=None)

        return series.astype(float)

    @classmethod
    def geometric(
        cls, series: Union[np.ndarray, pd.Series], decay: float = 0.5
    ) -> np.ndarray:
        """
        Applies geometric adstock to a time series.

        Parameters
        ----------
        series : np.ndarray or pd.Series
            Media spend time series
        decay : float
            Decay factor (0 < decay < 1)

        Returns
        -------
        np.ndarray
            Adstocked series
        """

        AdstockTransformer.logger.info(f"Applying geometric adstock with decay={decay}")

        cls.validate_decay(decay)
        series = cls.validate_series(series)

        result = np.zeros_like(series, dtype=float)

        for t in range(len(series)):
            if t == 0:
                result[t] = series[t]
            else:
                result[t] = series[t] + decay * result[t - 1]

        AdstockTransformer.logger.info("Adstock transformation completed")
        return result
