# src/models/mmm_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from src.evaluation.metrics import RegressionMetrics
from sklearn.model_selection import train_test_split


class RegularizedMMM:
    """
    Regularized Marketing Mix Model using Ridge Regression
    """

    def __init__(self, alpha: float = 1.0, test_size: float = 0.2, shuffle: bool = False, random_state=None):
        """
        Parameters
        ----------
        alpha : float
            Regularization strength
        test_size : float
            Fraction of data for test
        shuffle : bool
            Whether to shuffle data for train/test split
        """
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.coef_df = None

    def split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Chronological train/test split
        """
        return train_test_split(
            X, y,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train Ridge Regression
        """
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model.fit(X_train, y_train)

        self.coef_df = pd.DataFrame({
            "feature": X_train.columns,
            "coefficient": self.model.coef_
        }).sort_values(by="coefficient", ascending=False)

        return self

    def predict(self, X: pd.DataFrame):
        """
        Predict using trained Ridge model
        """
        return pd.Series(
            self.model.predict(X),
            index=X.index,
            name="predicted_sales"
        )

    def evaluate(self, X: pd.DataFrame = None, y: pd.Series = None):
        """
        Evaluate model on RMSE and R²
        """
        if X is None or y is None:
            X = self.X_test
            y = self.y_test

        y_pred = self.predict(X)
        metrics = RegressionMetrics.evaluate(y, y_pred)
        return metrics

    def get_coefficients(self):
        """
        Return feature coefficients
        """
        return self.coef_df

    def channel_contribution(self, channel_cols: list):
        """
        Calculate total contribution per channel
        """
        contributions = {}
        for col in channel_cols:
            idx = self.X_train.columns.get_loc(col)
            contributions[col] = self.model.coef_[idx] * self.X_train[col].sum()

        contrib_df = pd.DataFrame.from_dict(
            contributions, orient='index', columns=['total_contribution']
        ).sort_values(by='total_contribution', ascending=False)

        return contrib_df