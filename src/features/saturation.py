# src/features/saturation.py
import numpy as np


def hill_saturation(series: np.ndarray, alpha: float = 1, gamma: float = 0.5) -> np.ndarray:
    """
    Applies Hill saturation to a time series (diminishing returns).

    Parameters
    ----------
    series : np.ndarray
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
    return alpha * (series ** gamma) / ((series ** gamma) + (1 ** gamma))