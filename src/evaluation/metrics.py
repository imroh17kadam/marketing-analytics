import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RegressionMetrics:
    """
    Compute standard regression metrics
    """

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def r2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "RMSE": RegressionMetrics.rmse(y_true, y_pred),
            "MAE": RegressionMetrics.mae(y_true, y_pred),
            "R2": RegressionMetrics.r2(y_true, y_pred)
        }