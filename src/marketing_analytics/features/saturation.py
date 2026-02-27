from typing import Union

import numpy as np
import pandas as pd

from src.marketing_analytics.utils.logger import get_logger

logger = get_logger(__name__)


class SaturationTransformer:
    """
    Implements saturation (diminishing returns) functions for MMM.

    Currently supports:
    - Hill saturation
    """

    logger = get_logger(__qualname__)

    @staticmethod
    def validate_params(alpha: float, gamma: float) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0. Got: {alpha}")

        if gamma <= 0:
            raise ValueError(f"gamma must be > 0. Got: {gamma}")

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
            SaturationTransformer.logger.warning(
                "Input series contains NaN values — replacing with 0"
            )
            series = np.nan_to_num(series, nan=0.0)

        if (series < 0).any():
            SaturationTransformer.logger.warning(
                "Input series contains negative values — clipping to 0"
            )
            series = np.clip(series, a_min=0, a_max=None)

        return series.astype(float)

    @classmethod
    def hill(
        cls,
        series: Union[np.ndarray, pd.Series],
        alpha: float = 1.0,
        gamma: float = 0.5,
    ) -> np.ndarray:
        """
        Applies Hill saturation to a time series (diminishing returns).

        Parameters
        ----------
        series : np.ndarray or pd.Series
            Input series (usually adstocked)
        alpha : float
            Maximum effect
        gamma : float
            Half-saturation constant

        Returns
        -------
        np.ndarray
            Saturated series
        """

        SaturationTransformer.logger.info(
            f"Applying Hill saturation with alpha={alpha}, gamma={gamma}"
        )

        # Validate inputs
        cls.validate_params(alpha, gamma)
        series = cls.validate_series(series)

        # Hill saturation formula (your original logic preserved)
        saturated = alpha * (series**gamma) / ((series**gamma) + 1.0)

        SaturationTransformer.logger.info("Hill saturation transformation completed")
        return saturated
