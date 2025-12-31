# src/features/adstock.py
import numpy as np
import pandas as pd


def adstock_geometric(series: np.ndarray, decay: float = 0.5) -> np.ndarray:
    """
    Applies geometric adstock to a time series.

    Parameters
    ----------
    series : np.ndarray
        Media spend time series
    decay : float
        Decay factor (0 < decay < 1)

    Returns
    -------
    np.ndarray
        Adstocked series
    """
    result = np.zeros_like(series, dtype=float)
    for t in range(len(series)):
        if t == 0:
            result[t] = series[t]
        else:
            result[t] = series[t] + decay * result[t - 1]
    return result